"""Demo: python3 main.py [--plates N] [--seed S] [--offline]"""
import argparse

from data_loader import banner, describe, load_dataset, source_tag
from pipeline import Pipeline, format_batch, format_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plates", type=int, default=6, help="test plates to process")
    parser.add_argument("--seed", type=int, default=0, help="sample seed")
    parser.add_argument("--offline", action="store_true", help="do not try to download")
    args = parser.parse_args()

    dataset = load_dataset(download=not args.offline)
    pipe = Pipeline(dataset)
    print(describe(dataset, pipe.train, pipe.test))
    if dataset.source != "REAL":
        print("\nTo use the real data: download Faults.NNA from "
              "https://archive.ics.uci.edu/dataset/198/steel+plates+faults and place it in data/")
    print("\nThresholds from the training split:\n" + pipe.thresholds.render())
    reports = []
    for plate in pipe.sample(args.plates, args.seed):
        report = pipe.run_plate(plate)
        reports.append(report)
        print("\n[%s]\n%s" % (source_tag(dataset), format_report(report)))
    print("\n" + format_batch(*pipe.schedule(reports)))
    print("\n" + banner(dataset))


if __name__ == "__main__":
    main()
