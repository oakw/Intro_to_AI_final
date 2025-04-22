# Intro to Artificial Intelligence - Spring 2025
# Final Group Project - Explore and Exploit 2D Environment
# 
# Authors:
# Mārtiņš Prokuratovs
# Leo Saulītis
# 
import requests
from vizualizer import visualize_environment
import time
from datetime import datetime

API_PORT = 5000
API_BASE_URI = f"http://127.0.0.1{':' + str(API_PORT) if API_PORT is not None else ''}/1"
EXPLORE_ITERATIONS = 100
INITIAL_RANDOM_POINT_COUNT = 10
PATH_LENGTH = 10
ACQF_OPTIMIZATION_KWARGS = {'num_restarts': 20, 'raw_samples': 500}

class Agent:
    def __init__(self):
        self.moves = []

    def move_to(self, x: float, y: float): 
        """Record the move"""
        move = dict(x=x, y=y, z=self._query_z(x, y))
        if move['z'] is None:
            return None
        
        self.moves.append(move)
        return move

    def save_moves_csv(self, filename: str):
        """Save agent moves to a CSV file"""
        import csv
        with open(filename, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['x', 'y'])
            for move in self.moves:
                writer.writerow({key: move[key] for key in ['x', 'y']})
                
    def _query_z(self, x: float, y: float):
        """Query the actual z value at (x, y)"""
        response = requests.get(f"{API_BASE_URI}/{x}/{y}")
        time.sleep(0.1)  # small delay to avoid overwhelming the server
        
        if response.status_code == 200:
            return response.json().get('z')
        else:
            return None

class Explorer(Agent):
    def __init__(self, explore_class):
        """
        Initialize the explorer with an exploration class.
        Exploration class must have an 'explore' method.
        """
        super().__init__()
        self.explore_class = explore_class

    def start(self):
        """Start the exploration process"""
        if not hasattr(self.explore_class, 'explore'):
            raise ValueError("Explore class must have an 'explore' method")
        
        def query_z(x: float, y: float):
            """Accessor for querying the actual environment z value"""
            move = self.move_to(x, y)
            if move is None:
                return None
            return move['z']

        try:
            self.explore_class.explore(query_z)
        except Exception as e:
            print(f"Error during exploration: {e}")
            self.save_moves_csv(f"explore_error_{datetime.now().isoformat()}.csv")
            raise

    def save_moves_csv(self, filename: str = f"explore_{API_PORT}.csv"):
        """Save explorer moves to a CSV file"""
        super().save_moves_csv(filename)

class Exploiter(Agent):
    def __init__(self, explorer: Explorer|None = None):
        """
        Initialize the exploiter with an explorer instance.
        The exploiter will use the explorer's explore_class to exploit the environment.
        The explorer must have an 'explore_class' with an 'exploit' method.
        """
        super().__init__()
        if explorer is not None:
            if not hasattr(explorer.explore_class, 'exploit'):
                raise ValueError("Explorer class must have an 'exploit' method")
            
            self.exploiter_class = explorer.explore_class

    def start(self):
        def query_z(x: int, y: int):
            """Accessor for querying the actual environment z value"""
            move = self.move_to(int(x), int(y))
            if move is None:
                return None
            return move['z']
        
        self.exploiter_class.exploit(query_z)

    def save_moves_csv(self, filename: str = f"moves_{API_PORT}.csv"):
        """Save exploiter moves to a CSV file"""
        super().save_moves_csv(filename)

    def get_total_score(self):
        """Get the total score of the moves"""
        return sum(move['z'] for move in self.moves)

    def summary(self):
        """Print summary of moves"""
        print(f"\033[32m{self.exploiter_class.__class__.__name__}\033[0m Moves:")
        for move in self.moves:
            print(f"x: {move['x']}, y: {move['y']}, z: {move['z']}")

        print(f"Total z value: {self.get_total_score()}\n")


if __name__ == '__main__':
    from leo_sandbox.alternative import GaussianProcessExploreExploit
    from martins_sandbox.random_forest_regressor import RandomForestRegressorExploreExploit

    gp_explorer = Explorer(
        GaussianProcessExploreExploit(
            space_bounds=(-100, 100),
            explore_iterations=EXPLORE_ITERATIONS,
            initial_random_point_count=INITIAL_RANDOM_POINT_COUNT,
            path_length=PATH_LENGTH,
            acqf_optimization_kwargs={'num_restarts': 5, 'raw_samples': 20},
        )
    )
    gp_explorer.start()
    gp_explorer.save_moves_csv()
    explored_points = [(move.get('x'), move.get('y'), move.get('z')) for move in gp_explorer.moves]

    # Now run the Random Forest Regressor with the same explored points (to avoid full re-exploration)
    rf_explorer = Explorer(
        RandomForestRegressorExploreExploit(
            space_bounds=(-100, 100),
            n_iterations=0,
            path_length=PATH_LENGTH,
            predetermined_sample=explored_points,
        )
    )
    rf_explorer.start()

    # Choose whichever performed better in the exploitation phase
    gp_exploiter = Exploiter(gp_explorer)
    gp_exploiter.start()

    rf_exploiter = Exploiter(rf_explorer)
    rf_exploiter.start()

    gp_exploiter.summary()
    rf_exploiter.summary()

    if gp_exploiter.get_total_score() > rf_exploiter.get_total_score():
        print("GP exploiter is better")
        gp_exploiter.save_moves_csv()
        visualize_environment(gp_exploiter.exploiter_class)
    else:
        print("RF exploiter is better")
        rf_exploiter.save_moves_csv()
        visualize_environment(rf_exploiter.exploiter_class)

