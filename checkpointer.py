import pickle

class Cornifer():
    @staticmethod
    def save(data, filename, suffix=None):
        with open(f'{filename}_{suffix}.pkl', "wb") as f:
            pickle.dump(data, f)

    @staticmethod
    def checkpoint(data, filename, epoch, suffix=None):
        with open(f'{filename}_{suffix}.pkl', "wb") as f:
            pickle.dump(data, f)
        print(f"Cornifer: checkpoint made for epoch: {epoch}")
    
    @staticmethod
    def load(filename, suffix):
        try:
            with open(f'{filename}_{suffix}.pkl', "rb") as f:
                data = pickle.load(f)
            print("Checkpoint retrieved!")
            return data
        except FileNotFoundError:
            print("Cornifer: sorry boss, no saved data found.") 
        return None



