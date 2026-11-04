"""Read metrics from an offline W&B run's .wandb file, with no network. Prints JSONL.

The file is a 7-byte header, then LevelDB-log framed protobuf records: 32 KiB blocks, each record CRC(4) length(2) type(1),
type 1 full, 2 first, 3 middle, 4 last.
"""
import glob
import json
import struct
import sys

from wandb.proto import wandb_internal_pb2 as pb

BLOCK, HEADER, FILE_HEADER = 32768, 7, 7


def payloads(path):
    data = open(path, "rb").read()
    position, pending = FILE_HEADER, b""
    while position + HEADER <= len(data):
        room = BLOCK - (position % BLOCK)
        if room < HEADER:
            position += room
            continue
        _crc, length, kind = struct.unpack("<IHB", data[position : position + HEADER])
        if position + HEADER + length > len(data):  # torn at the end of a killed run
            return
        chunk = data[position + HEADER : position + HEADER + length]
        position += HEADER + length
        if kind == 1:
            yield chunk
        elif kind == 2:
            pending = chunk
        elif kind == 3:
            pending += chunk
        elif kind == 4:
            yield pending + chunk
        elif kind == 0 and length == 0:  # zero padding: move to the next block
            position += BLOCK - (position % BLOCK) if position % BLOCK else 0


def history(path):
    for payload in payloads(path):
        record = pb.Record()
        record.ParseFromString(payload)
        if record.WhichOneof("record_type") == "history":
            yield {item.key or "/".join(item.nested_key): json.loads(item.value_json) for item in record.history.item}


if __name__ == "__main__":
    for root in sys.argv[1:]:
        for path in sorted(glob.glob(root + "/wandb/offline-run-*/run-*.wandb")):
            for row in history(path):
                row["_file"] = path.split("/")[-2]
                print(json.dumps(row))
