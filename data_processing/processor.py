import pandas as pd
from sklearn.utils import shuffle
from typing import Tuple, List
import logging
import re
from Bio import SeqIO

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class SequenceDataProcessor:
    VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")

    def __init__(self, file_path: str):
        self.file_path = file_path
        if file_path.endswith(".fasta") or file_path.endswith(".fa"):
            self.df = self.fasta_to_dataframe(file_path)
        else:
            self.df = pd.read_csv(file_path)
        logging.info(f"Loaded data from {file_path}")

    @staticmethod
    def fasta_to_dataframe(fasta_file: str) -> pd.DataFrame:
        """Convert a FASTA file to a pandas DataFrame."""
        sequences = []
        names = []
        for record in SeqIO.parse(fasta_file, "fasta"):
            sequences.append(str(record.seq))
            names.append(record.id)
        return pd.DataFrame({"Name": names, "Sequence": sequences})

    def remove_duplicates(self) -> None:
        """Remove duplicate sequences from the dataset."""
        initial_count = len(self.df)
        self.df.drop_duplicates(subset="Sequence", inplace=True)
        final_count = len(self.df)
        duplicate_count = initial_count - final_count
        logging.info(f"Removed {duplicate_count} duplicate sequences.")

    def remove_long_sequences(self, max_length: int = 100) -> None:
        """Remove sequences longer than the specified maximum length."""
        initial_count = len(self.df)
        self.df = self.df[self.df["Sequence"].str.len() <= max_length]
        removed_count = initial_count - len(self.df)
        logging.info(
            f"Removed {removed_count} sequences longer than {max_length} characters."
        )

    def validate_amino_acid_sequences(self) -> None:
        """Remove sequences containing invalid amino acid characters."""
        initial_count = len(self.df)
        valid_sequence = self.df["Sequence"].apply(
            lambda x: set(x).issubset(self.VALID_AMINO_ACIDS)
        )
        self.df = self.df[valid_sequence]
        removed_count = initial_count - len(self.df)
        logging.info(
            f"Removed {removed_count} sequences with invalid amino acid characters."
        )

    def concatenate_datasets(self, other_file_paths: List[str]) -> None:
        """Concatenate multiple datasets."""
        dfs = [self.df]
        for path in other_file_paths:
            if path.endswith(".fasta") or path.endswith(".fa"):
                dfs.append(self.fasta_to_dataframe(path))
            else:
                dfs.append(pd.read_csv(path))
        self.df = pd.concat(dfs).reset_index(drop=True)
        logging.info(f"Concatenated {len(dfs)} datasets. New shape: {self.df.shape}")

    def shuffle_data(self) -> None:
        """Shuffle the dataset."""
        self.df = shuffle(self.df).reset_index(drop=True)
        logging.info("Dataset shuffled.")

    def split_data(
        self, ratios: Tuple[float, float, float] = (0.05, 0.05, 0.9)
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split the data into validation, test, and train sets."""
        assert sum(ratios) == 1, "Ratios must sum to 1"
        total_len = len(self.df)
        valid_idx = int(total_len * ratios[0])
        test_idx = valid_idx + int(total_len * ratios[1])

        valid_df = self.df[:valid_idx]
        test_df = self.df[valid_idx:test_idx]
        train_df = self.df[test_idx:]

        logging.info(
            f"Data split into: Valid ({len(valid_df)}), Test ({len(test_df)}), Train ({len(train_df)})"
        )
        return valid_df, test_df, train_df

    def save_data(self, output_path: str) -> None:
        """Save the processed data to a CSV file."""
        self.df.to_csv(output_path, index=False)
        logging.info(f"Saved processed data to {output_path}")


def main():
    # You can now use either FASTA or CSV files as input
    processor = SequenceDataProcessor("input.fasta")  # or 'input.csv'

    processor.remove_duplicates()
    processor.remove_long_sequences()
    processor.validate_amino_acid_sequences()
    processor.concatenate_datasets(["other_data.fasta", "more_data.csv"])
    processor.shuffle_data()

    valid_df, test_df, train_df = processor.split_data()

    valid_df.to_csv("valid_general.csv", index=False)
    test_df.to_csv("test_general.csv", index=False)
    train_df.to_csv("train_general.csv", index=False)

    processor.save_data("processed_data.csv")


if __name__ == "__main__":
    main()
