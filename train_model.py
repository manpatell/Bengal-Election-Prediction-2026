from __future__ import annotations

from pprint import pprint

from bengal_election.pipeline import train_and_save_model


def main() -> None:
    bundle = train_and_save_model()
    print("Training completed successfully.")
    print("Saved artifact: models/bengal_election_bundle.pkl")
    print("Validation summary:")
    pprint(bundle["metadata"]["metrics"])


if __name__ == "__main__":
    main()
