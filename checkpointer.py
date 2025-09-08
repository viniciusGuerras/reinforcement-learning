import pickle
from typing import Any

class Cornifer():
    @staticmethod
    def save(data: Any, filename: str, suffix: str | None = None) -> None:
        with open(f'{filename}_{suffix}.pkl', "wb") as f:
            pickle.dump(data, f)

    @staticmethod
    def checkpoint(data: Any, filename: str, epoch: int, suffix: str | None = None) -> None:
        with open(f'{filename}_{suffix}.pkl', "wb") as f:
            pickle.dump(data, f)
        print(f"Cornifer: checkpoint made for epoch: {epoch}")
    
    @staticmethod
    def load(filename: str, suffix: str) -> Any | None:
        try:
            with open(f'{filename}_{suffix}.pkl', "rb") as f:
                data = pickle.load(f)
            print("Checkpoint retrieved!")
            return data
        except FileNotFoundError:
            print("Cornifer: sorry boss, no saved data found.") 
        return None



