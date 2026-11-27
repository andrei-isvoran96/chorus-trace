"""Train and select random-initialized checkpoints using original-mix validation."""

import math
import tempfile
import wave
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .data import Example
from .fulltrack_data import FullTrack
from .fulltrack_evaluation import evaluate_full_tracks
from .model import DEFAULT_DILATIONS
from .phonemes import PhonemeSupervision
from .recognizer import TrainingConfig, train


def _identity_keys(track: FullTrack):
    keys = (
        *(("artist", artist) for artist in track.artist_ids),
        ("song", track.song_id),
        ("lyrics", track.lyric_id),
        ("recording", track.recording_family_id),
    )
    return keys + (("text", track.text),) if track.text else keys


def _training_groups(tracks: Sequence[FullTrack], seed: int) -> list[list[FullTrack]]:
    """Group linked artists and alternate versions before shuffling groups."""
    tracks = sorted(tracks, key=lambda track: track.track_id)
    parent = list(range(len(tracks)))
    owners = {}

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for index, track in enumerate(tracks):
        for key in _identity_keys(track):
            first = owners.setdefault(key, index)
            parent[root(index)] = root(first)

    groups = {}
    for index, track in enumerate(tracks):
        groups.setdefault(root(index), []).append(track)
    group_list = list(groups.values())
    return [group_list[index] for index in np.random.default_rng(seed).permutation(len(groups))]


def _clip_spans(track: FullTrack, clip_seconds: float):
    """Use training word times to cut complete words and label instrumental spans."""
    words = track.words
    if not words and track.condition != "instrumental":
        raise ValueError(f"{track.track_id}: training requires timed lyric words")
    first = 0
    for last, word in enumerate(words):
        following_start = words[last + 1].start if last + 1 < len(words) else track.duration_seconds
        if last + 1 < len(words) and following_start - word.end < 0.8:
            if words[last + 1].end - words[first].start <= clip_seconds:
                continue
        previous_end = words[first - 1].end if first else 0.0
        start = max(words[first].start - 0.12, (previous_end + words[first].start) / 2)
        end = min(word.end + 0.12, (word.end + following_start) / 2)
        yield start, end, " ".join(item.word for item in words[first:last + 1])
        first = last + 1

    for start, end in track.instrumental_intervals or ():
        duration = end - start
        if duration < 0.3:
            continue
        count = math.ceil(duration / clip_seconds)
        for index in range(count):
            yield start + duration * index / count, start + duration * (index + 1) / count, ""


def _prepare_clips(track: FullTrack, output: Path, clip_seconds: float) -> list[Example]:
    if not track.full_track:
        if track.split != "train" or track.duration_seconds > clip_seconds:
            raise ValueError(f"{track.track_id}: prepared clips must be training audio of at most {clip_seconds:g} seconds")
        return [Example(track.audio, track.text, track.song_id, "train", track.condition, track.track_id)]

    output.mkdir()
    examples = []
    with wave.open(str(track.audio), "rb") as source:
        rate = source.getframerate()
        for index, (start, end, text) in enumerate(_clip_spans(track, clip_seconds)):
            first = math.floor(start * rate)
            last = min(source.getnframes(), math.ceil(end * rate))
            source.setpos(first)
            audio = output / f"{index}.wav"
            with wave.open(str(audio), "wb") as clip:
                clip.setnchannels(source.getnchannels())
                clip.setsampwidth(source.getsampwidth())
                clip.setframerate(rate)
                clip.writeframes(source.readframes(last - first))
            examples.append(Example(audio, text, track.song_id, "train", track.condition))
    return examples


def train_full_tracks(
    tracks: Sequence[FullTrack],
    checkpoint: str | Path,
    *,
    config: TrainingConfig | None = None,
    scale_groups: Sequence[int] = (),
    clip_seconds: float = 12.0,
    validate_every: int = 1,
    phonemes: PhonemeSupervision | None = None,
    test_lock_sha256: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict:
    """Fit nested training groups, choose on original mixes, and save the winner."""
    if not math.isfinite(clip_seconds) or clip_seconds <= 0 or validate_every < 1:
        raise ValueError("clip seconds and validation interval must be positive")
    if any(track.split not in {"train", "validation"} for track in tracks):
        raise ValueError("full-track development accepts only train and validation tracks")
    training = [track for track in tracks if track.split == "train"]
    validation = [track for track in tracks if track.split == "validation"]
    if not training or not validation:
        raise ValueError("full-track development requires train and validation tracks")
    if any(track.condition != "original_mix" for track in validation):
        raise ValueError("validation requires original-mix audio")
    training_keys = {key for track in training for key in _identity_keys(track)}
    if any(training_keys.intersection(_identity_keys(track)) for track in validation):
        raise ValueError("train and validation share an artist, song, lyrics, or recording family")

    config = config or TrainingConfig()
    groups = _training_groups(training, config.seed)
    group_count = len(groups)
    scales = scale_groups or (max(1, group_count // 4), max(1, group_count // 2), group_count)
    if any(not isinstance(scale, int) or scale < 1 or scale > group_count for scale in scales):
        raise ValueError(f"training scales must be counts between 1 and {group_count}")
    scales = sorted(set(scales))
    learning_curve = []
    best_key = None
    best_selection = None
    best_recognizer = None
    with tempfile.TemporaryDirectory(prefix="chorustrace-training-") as directory:
        active_tracks = [track for group in groups[:scales[-1]] for track in group]
        clips = {
            track.track_id: _prepare_clips(track, Path(directory) / str(index), clip_seconds)
            for index, track in enumerate(active_tracks)
        }
        for size in scales:
            selected = [track for group in groups[:size] for track in group]
            examples = [clip for track in selected for clip in clips[track.track_id]]
            history = []

            def record_loss(epoch, loss):
                history.append({"epoch": epoch, "mean_training_loss": loss})

            def validate(epoch, recognizer):
                if epoch % validate_every and epoch != config.epochs:
                    return None
                metrics = evaluate_full_tracks(recognizer, validation, samples=0)["aggregate"]
                history[-1]["validation"] = metrics
                wer, cer = metrics["wer"], metrics["cer"]
                if wer is None or cer is None:
                    raise ValueError("validation tracks need nonempty references")
                if progress is not None:
                    progress(f"{size} training groups, epoch {epoch}, validation WER {wer:.2%}, CER {cer:.2%}")
                return wer, cer

            recognizer = train(
                examples, config, progress=record_loss, validation_score=validate,
                feature_cache_dir=directory, phonemes=phonemes,
            )
            scored = [epoch for epoch in history if "validation" in epoch]

            def key(epoch):
                return epoch["validation"]["wer"], epoch["validation"]["cer"]

            best_epoch = min(scored, key=key)
            learning_curve.append({
                "training_groups": size,
                "training_tracks": len(selected),
                "training_audio_seconds": sum(track.duration_seconds for track in selected),
                "training_clips": len(examples),
                "best_epoch": best_epoch["epoch"],
                "epochs": history,
            })
            if phonemes is not None:
                learning_curve[-1]["phoneme_supervised_clips"] = sum(
                    example.clip_id in phonemes.targets for example in examples
                )
            score = key(best_epoch)
            if best_key is None or score < best_key:
                best_key = score
                best_selection = {"training_groups": size, "epoch": best_epoch["epoch"]}
                best_recognizer = recognizer
    validation_report = evaluate_full_tracks(best_recognizer, validation, compare_unsegmented=True)
    best_recognizer.test_lock_sha256 = test_lock_sha256
    best_recognizer.save(checkpoint)
    configuration = asdict(config)
    if not config.features.stereo_features:
        del configuration["features"]["stereo_features"]
    if config.dilations == DEFAULT_DILATIONS:
        del configuration["dilations"]
    if not config.input_deltas:
        del configuration["input_deltas"]
    if not config.ctc_prior_weight:
        del configuration["ctc_prior_weight"]
    if not config.phoneme_loss_weight:
        del configuration["phoneme_loss_weight"]
    report = {
        "configuration": configuration,
        "training_clip_seconds": clip_seconds,
        "validate_every": validate_every,
        "selection": best_selection,
        "learning_curve": learning_curve,
        "validation": validation_report,
    }
    if phonemes is not None:
        report["phoneme_supervision"] = {"sha256": phonemes.sha256, "symbols": len(phonemes.vocabulary)}
    excerpts = sorted(track.track_id for track in validation if not track.full_track)
    if excerpts:
        report["validation_scope"] = {
            "condition": "original_mix",
            "full_tracks": len(validation) - len(excerpts),
            "excerpt_tracks": len(excerpts),
            "excerpt_track_ids": excerpts,
        }
    return report
