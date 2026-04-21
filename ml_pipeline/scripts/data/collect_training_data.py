import argparse
import csv
import json
from confluent_kafka import Consumer, KafkaException


def parse_args():
    parser = argparse.ArgumentParser(description="Collect labeled SAGE training rows from Kafka.")
    parser.add_argument("--topic", default="gateway-telemetry", help="Kafka topic to consume")
    parser.add_argument("--bootstrap-servers", default="localhost:9092", help="Kafka bootstrap server(s)")
    parser.add_argument("--group-id", default="training-data-export", help="Kafka consumer group id")
    parser.add_argument("--output", default="ml_pipeline/data/training_data.csv", help="Output CSV path")
    parser.add_argument("--max-rows", type=int, default=20000, help="Maximum number of rows to export")
    parser.add_argument("--consumer-timeout-ms", type=int, default=10000, help="Stop if no new messages during timeout")
    return parser.parse_args()


def build_row(event):
    features = event.get("features", {}) if isinstance(event, dict) else {}
    label = (event.get("label") if isinstance(event, dict) else None) or "unknown"

    return [
        float(features.get("sessionDepth", 0.0) or 0.0),
        float(features.get("temporalVariance", 0.0) or 0.0),
        float(features.get("requestVelocity", 0.0) or 0.0),
        float(features.get("behavioralDiversity", 0.0) or 0.0),
        float(features.get("endpointConcentration", 0.0) or 0.0),
        float(features.get("cartRatio", 0.0) or 0.0),
        float(features.get("assetSkipRatio", 0.0) or 0.0),
        float(features.get("sequentialTraversal", 0.0) or 0.0),
        label,
    ]


def main():
    args = parse_args()

    conf = {
        "bootstrap.servers": args.bootstrap_servers,
        "group.id": args.group_id,
        "auto.offset.reset": "latest",
    }
    consumer = Consumer(conf)
    consumer.subscribe([args.topic])

    headers = [
        "SAGE_Session_Depth",
        "SAGE_Temporal_Variance",
        "SAGE_Request_Velocity",
        "SAGE_Behavioral_Diversity",
        "SAGE_Endpoint_Concentration",
        "SAGE_Cart_Ratio",
        "SAGE_Asset_Skip_Ratio",
        "SAGE_Sequential_Traversal",
        "label",
    ]

    rows_written = 0
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        try:
            while rows_written < args.max_rows:
                msg = consumer.poll(timeout=args.consumer_timeout_ms / 1000.0)
                if msg is None:
                    break
                if msg.error():
                    raise KafkaException(msg.error())

                payload = msg.value().decode("utf-8")
                event = json.loads(payload)
                row = build_row(event)
                writer.writerow(row)
                rows_written += 1
        finally:
            consumer.close()

    print(f"Export complete. Rows written: {rows_written}")
    print(f"Output file: {args.output}")


if __name__ == "__main__":
    main()
