import os
import re

# Set the path to your directory here
directory_path = './results' 

def count_wins(directory):
    player_a_wins = 0
    player_b_wins = 0
    
    # Regex to capture the name before "wins by"
    win_pattern = re.compile(r"(.+)\s+wins\s+\d+")

    if not os.path.exists(directory):
        print(f"Error: The directory '{directory}' does not exist.")
        return

    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            try:
                # ADDED: encoding='utf-8' and errors='ignore'
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                    lines = file.readlines()
                    
                    if len(lines) >= 4:
                        target_line = lines[-4].strip()

                        if "PLAYER_A" in target_line:
                            player_a_wins += 1
                        elif "PLAYER_B" in target_line:
                            player_b_wins += 1
            
            except Exception as e:
                print(f"Could not read file {filename}: {e}")

    print("--- Results ---")
    print(f"PLAYER_A Total Wins: {player_a_wins}")
    print(f"PLAYER_B Total Wins: {player_b_wins}")

if __name__ == "__main__":
    count_wins(directory_path)