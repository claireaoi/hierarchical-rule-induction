'''To check highway checkpoints for one model'''
import argparse
import pdb
import os
from os import listdir
from os.path import isfile, join


parser = argparse.ArgumentParser()
parser.add_argument('--log_path', type=str, 
    help='File path for txt log files. You need to run learn-ppo.py with test-only to obtain those txt files before.')
# parser.add_argument('--save_path', type=str, default=None, 
#     help='File path for analysis files.')
# parser.add_argument('--test_car_number', type=int,
parser.add_argument('--test_car_number', type=int,
    help='Check logs obtained by testing a certain number of cars.')
args = parser.parse_args()


def log_summary(src, trained_epoch, test_car, mode):

    with open(src, "r+", encoding="utf-8") as file:
        contents = file.readlines()
        length_list = []
        succ_list = []
        reward_list = []
        avg_speed_list = []
        distance_list = []
        accelerate_list = []
        decelerate_list = []
        turn_left_list = []
        turn_right_list = []

        for line in contents:
            line = line.strip().split(",")
            length = int(line[0])
            succ = int(line[1])
            reward = float(line[2])
            avg_speed = float(line[3])
            distance = float(line[4])
            accelerate = int(line[5])
            decelerate = int(line[6])
            turn_left = int(line[7])
            turn_right = int(line[8])

            length_list.append(length)
            succ_list.append(succ)
            reward_list.append(reward)
            avg_speed_list.append(avg_speed)
            distance_list.append(distance)
            accelerate_list.append(accelerate)
            decelerate_list.append(decelerate)
            turn_left_list.append(turn_left)
            turn_right_list.append(turn_right)

    episode = len(length_list)
    print(f"Total episodes: {episode}")

    avg_length = sum(length_list) / episode
    succ_rate = sum(succ_list) / episode
    avg_reward = sum(reward_list) / episode
    avg_speed = sum(avg_speed_list) / episode
    avg_distance = sum(distance_list) / episode
    avg_accelerate = sum(accelerate_list) / episode
    avg_decelerate = sum(decelerate_list) / episode
    avg_turn_left = sum(turn_left_list) / episode
    avg_turn_right = sum(turn_right_list) / episode
    print(f"-" * 30)
    print(f"[*] Average length: {avg_length}")
    print(f"[*] Succ rate: {succ_rate}")
    print(f"[*] Average reward: {avg_reward}")
    print(f"[*] Average speed: {avg_speed}")
    print(f"[*] Average distance: {avg_distance}")
    print(f"[*] Average acceleration: {avg_accelerate}")
    print(f"[*] Average deceleration: {avg_decelerate}")
    print(f"[*] Average left turn: {avg_turn_left}")
    print(f"[*] Average right turn: {avg_turn_right}")
    print(f"-" * 30)


def main():
    # read all txt files then summarize and give the best one.
    # save_path = args.save_path if args.save_path is not None else args.log_path
    # save_path = join(save_path, 'summary.txt')
    # print(f'=' * 10)
    # print(f'save summary to {save_path}')
    # print(f'=' * 10)
    for file in os.listdir(args.log_path):
        if file.endswith(".txt"):
            info = file.split('_')
            trained_epoch = int(info[0])
            test_car = int(info[1])
            mode = info[2][:-4]
            if test_car != args.test_car_number:
                continue
            print(f'=' * 10)
            print(f'test_trained_epoch: {trained_epoch}, test_car_num: {test_car}, eval_mode: {mode}')
            print(f'=' * 10)
            log_summary(join(args.log_path, file), trained_epoch, test_car, mode)
            

if __name__ == '__main__':
    main()
