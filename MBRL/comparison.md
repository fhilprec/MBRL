# old training function

### MODEL_ARCHITECTURE = LSTM
Epoch 200/2000, Loss: 0.028649, LR: 1.96e-04
Epoch 400/2000, Loss: 0.020004, LR: 1.83e-04
Epoch 600/2000, Loss: 0.016011, LR: 1.63e-04
Epoch 800/2000, Loss: 0.013681, LR: 1.38e-04
Epoch 1000/2000, Loss: 0.012077, LR: 1.10e-04
Epoch 1200/2000, Loss: 0.011015, LR: 8.23e-05
Epoch 1400/2000, Loss: 0.010265, LR: 5.72e-05
Epoch 1600/2000, Loss: 0.009769, LR: 3.73e-05
Epoch 1800/2000, Loss: 0.009441, LR: 2.44e-05
Epoch 2000/2000, Loss: 0.009199, LR: 2.00e-05

### MODEL_ARCHITECTURE = V2_LSTM
Epoch 200/2000, Loss: 0.020443, LR: 1.96e-04
Epoch 400/2000, Loss: 0.013689, LR: 1.83e-04
Epoch 600/2000, Loss: 0.009699, LR: 1.63e-04
Epoch 800/2000, Loss: 0.007063, LR: 1.38e-04
Epoch 1000/2000, Loss: 0.005456, LR: 1.10e-04
Epoch 1200/2000, Loss: 0.004434, LR: 8.23e-05
Epoch 1400/2000, Loss: 0.003808, LR: 5.72e-05
Epoch 1600/2000, Loss: 0.003443, LR: 3.73e-05
Epoch 1800/2000, Loss: 0.003227, LR: 2.44e-05
Epoch 2000/2000, Loss: 0.003075, LR: 2.00e-05

# new training function


### MODEL_ARCHITECTURE = LSTM


### MODEL_ARCHITECTURE = V2_LSTM
Epoch 200/2000, Loss: 0.032467, LR: 1.96e-04
Epoch 400/2000, Loss: 0.016122, LR: 1.83e-04
Epoch 600/2000, Loss: 0.010290, LR: 1.63e-04
Epoch 800/2000, Loss: 0.006772, LR: 1.38e-04
Epoch 1000/2000, Loss: 0.004981, LR: 1.10e-04
Epoch 1200/2000, Loss: 0.003921, LR: 8.23e-05
Epoch 1400/2000, Loss: 0.003299, LR: 5.72e-05
Epoch 1600/2000, Loss: 0.002939, LR: 3.73e-05
Epoch 1800/2000, Loss: 0.002721, LR: 2.44e-05
Epoch 2000/2000, Loss: 0.002586, LR: 2.00e-05

### MODEL_ARCHITECTURE = V2_LSTM withouts weights
Epoch 200/2000, Loss: 0.032467, LR: 1.96e-04
Epoch 400/2000, Loss: 0.016122, LR: 1.83e-04
Epoch 600/2000, Loss: 0.010290, LR: 1.63e-04
Epoch 800/2000, Loss: 0.006772, LR: 1.38e-04
Epoch 1000/2000, Loss: 0.004981, LR: 1.10e-04
Epoch 1200/2000, Loss: 0.003921, LR: 8.23e-05
Epoch 1400/2000, Loss: 0.003299, LR: 5.72e-05
Epoch 1600/2000, Loss: 0.002939, LR: 3.73e-05
Epoch 1800/2000, Loss: 0.002721, LR: 2.44e-05
Epoch 2000/2000, Loss: 0.002586, LR: 2.00e-05

V5,6,7 all worse

### MODEL_ARCHITECTUE = V2_LSTM without weights
Epoch 200/2000, Loss: 0.015432, LR: 1.96e-04
Epoch 400/2000, Loss: 0.006254, LR: 1.83e-04
Epoch 600/2000, Loss: 0.002574, LR: 1.63e-04
Epoch 800/2000, Loss: 0.001375, LR: 1.38e-04
Epoch 1000/2000, Loss: 0.000856, LR: 1.10e-04
Epoch 1200/2000, Loss: 0.000617, LR: 8.23e-05
Epoch 1400/2000, Loss: 0.000489, LR: 5.72e-05
Epoch 1600/2000, Loss: 0.000416, LR: 3.73e-05
Epoch 1800/2000, Loss: 0.000372, LR: 2.44e-05
Epoch 2000/2000, Loss: 0.000342, LR: 2.00e-05







Best one V2
MBRLfhilprec@Florian:~/MBRL$ python MBRL/worldmodel.py V2 render verbos
pygame 2.6.1 (SDL 2.28.4, Python 3.10.17)
Hello from the pygame community. https://www.pygame.org/contribute.html
Loading existing experience data from experience_data_LSTM.pkl...
Epoch 2000/20000, Loss: 0.000101, LR: 1.96e-04
[552, 617, 1071, 1209, 1269, 1331, 1386, 1449, 1589, 1972, 2096, 2199, 2257, 2387, 2449, 2505, 2851]
Training on 2387 states...
Loading existing model from world_model_V2_LSTM.pkl...
State reset
Step 0, Unnormalized Error: 0.65 | Action: DOWN
Step 1, Unnormalized Error: 0.31 | Action: DOWN
Step 2, Unnormalized Error: 0.29 | Action: UPLEFTFIRE
Step 3, Unnormalized Error: 0.18 | Action: DOWNLEFTFIRE
Step 4, Unnormalized Error: 0.16 | Action: UPFIRE
Step 5, Unnormalized Error: 0.16 | Action: UPFIRE
Step 6, Unnormalized Error: 0.12 | Action: DOWN
Step 7, Unnormalized Error: 0.12 | Action: DOWN
Step 8, Unnormalized Error: 0.12 | Action: LEFT
Step 9, Unnormalized Error: 0.13 | Action: UPLEFT
Step 10, Unnormalized Error: 0.13 | Action: DOWNRIGHT
Step 11, Unnormalized Error: 0.11 | Action: DOWN
Step 12, Unnormalized Error: 0.11 | Action: DOWN
Step 13, Unnormalized Error: 0.09 | Action: DOWN
Step 14, Unnormalized Error: 0.09 | Action: DOWNRIGHT
Step 15, Unnormalized Error: 0.08 | Action: UPRIGHT
Step 16, Unnormalized Error: 0.08 | Action: RIGHTFIRE
Step 17, Unnormalized Error: 0.09 | Action: LEFTFIRE
Step 18, Unnormalized Error: 0.09 | Action: UP
Step 19, Unnormalized Error: 0.09 | Action: UP
State reset
Step 20, Unnormalized Error: 0.01 | Action: DOWNRIGHT
Step 21, Unnormalized Error: 0.01 | Action: DOWN
Step 22, Unnormalized Error: 0.02 | Action: NOOP
Step 23, Unnormalized Error: 0.02 | Action: UPRIGHTFIRE
Step 24, Unnormalized Error: 0.02 | Action: UPRIGHT
Step 25, Unnormalized Error: 0.01 | Action: RIGHT
Step 26, Unnormalized Error: 0.01 | Action: DOWN
Step 27, Unnormalized Error: 0.02 | Action: FIRE
Step 28, Unnormalized Error: 0.04 | Action: UPRIGHT
Step 29, Unnormalized Error: 0.02 | Action: DOWNRIGHT
Step 30, Unnormalized Error: 0.04 | Action: UPLEFT
Step 31, Unnormalized Error: 0.06 | Action: DOWNLEFT
Step 32, Unnormalized Error: 0.25 | Action: DOWNRIGHT
Step 33, Unnormalized Error: 0.49 | Action: DOWNRIGHTFIRE
Step 34, Unnormalized Error: 0.88 | Action: DOWNRIGHTFIRE
Step 35, Unnormalized Error: 12.88 | Action: DOWNFIRE
Step 36, Unnormalized Error: 57.01 | Action: UP
Step 37, Unnormalized Error: 40.22 | Action: DOWNLEFT
Step 38, Unnormalized Error: 30.10 | Action: DOWN
Step 39, Unnormalized Error: 41.37 | Action: UPFIRE
State reset
Step 40, Unnormalized Error: 0.79 | Action: DOWNLEFT
Step 41, Unnormalized Error: 0.26 | Action: UPRIGHT
Step 42, Unnormalized Error: 0.11 | Action: UP
Step 43, Unnormalized Error: 0.60 | Action: FIRE
Step 44, Unnormalized Error: 1.70 | Action: DOWNFIRE
Step 45, Unnormalized Error: 2.82 | Action: RIGHTFIRE
Step 46, Unnormalized Error: 2.91 | Action: UPRIGHTFIRE
Step 47, Unnormalized Error: 25.36 | Action: UPLEFTFIRE
Step 48, Unnormalized Error: 13.47 | Action: UPRIGHTFIRE
Step 49, Unnormalized Error: 22.45 | Action: DOWNRIGHT
Step 50, Unnormalized Error: 46.63 | Action: NOOP
Step 51, Unnormalized Error: 245.42 | Action: UPRIGHTFIRE
Step 52, Unnormalized Error: 167.72 | Action: UPLEFTFIRE
Step 53, Unnormalized Error: 154.00 | Action: LEFTFIRE
Step 54, Unnormalized Error: 141.56 | Action: DOWNFIRE
Step 55, Unnormalized Error: 132.28 | Action: DOWN
Step 56, Unnormalized Error: 13.25 | Action: UPLEFTFIRE
Step 57, Unnormalized Error: 20.94 | Action: DOWN
Step 58, Unnormalized Error: 57.36 | Action: NOOP
Step 59, Unnormalized Error: 40.32 | Action: NOOP
State reset
Step 60, Unnormalized Error: 9.93 | Action: FIRE
Step 61, Unnormalized Error: 7.41 | Action: DOWN
Step 62, Unnormalized Error: 34.42 | Action: DOWN
Step 63, Unnormalized Error: 5.97 | Action: DOWN
Step 64, Unnormalized Error: 12.15 | Action: DOWNFIRE
Step 65, Unnormalized Error: 17.84 | Action: LEFT
Step 66, Unnormalized Error: 29.70 | Action: DOWNLEFT
Step 67, Unnormalized Error: 51.47 | Action: UP
Step 68, Unnormalized Error: 59.34 | Action: DOWN
Step 69, Unnormalized Error: 13.53 | Action: UPFIRE
Step 70, Unnormalized Error: 14.42 | Action: UPLEFTFIRE
Step 71, Unnormalized Error: 25.92 | Action: LEFT
Step 72, Unnormalized Error: 56.82 | Action: RIGHTFIRE
Step 73, Unnormalized Error: 19.36 | Action: DOWN
Step 74, Unnormalized Error: 19.74 | Action: UPFIRE
Step 75, Unnormalized Error: 20.74 | Action: LEFTFIRE
Step 76, Unnormalized Error: 25.25 | Action: UP
Step 77, Unnormalized Error: 32.10 | Action: UPLEFT
Step 78, Unnormalized Error: 34.07 | Action: DOWN
Step 79, Unnormalized Error: 37.31 | Action: UPRIGHT
State reset
Step 80, Unnormalized Error: 5.68 | Action: DOWN
Step 81, Unnormalized Error: 1.43 | Action: UPRIGHTFIRE
Step 82, Unnormalized Error: 1.06 | Action: FIRE
Step 83, Unnormalized Error: 1.02 | Action: DOWN
Step 84, Unnormalized Error: 0.99 | Action: DOWNRIGHTFIRE
Step 85, Unnormalized Error: 0.92 | Action: RIGHTFIRE
Step 86, Unnormalized Error: 0.84 | Action: DOWN
Step 87, Unnormalized Error: 0.76 | Action: UPRIGHT
Step 88, Unnormalized Error: 2.46 | Action: LEFTFIRE
Step 89, Unnormalized Error: 0.59 | Action: UPLEFTFIRE
Step 90, Unnormalized Error: 0.55 | Action: DOWN
Step 91, Unnormalized Error: 0.60 | Action: RIGHTFIRE
Step 92, Unnormalized Error: 0.54 | Action: UPLEFTFIRE
Step 93, Unnormalized Error: 0.44 | Action: DOWN
Step 94, Unnormalized Error: 0.47 | Action: UPLEFTFIRE
Step 95, Unnormalized Error: 0.47 | Action: DOWN
Step 96, Unnormalized Error: 0.65 | Action: LEFT
Step 97, Unnormalized Error: 0.64 | Action: NOOP
Step 98, Unnormalized Error: 0.61 | Action: FIRE
Step 99, Unnormalized Error: 0.78 | Action: DOWNFIRE
State reset
Step 100, Unnormalized Error: 0.07 | Action: DOWNRIGHT
Step 101, Unnormalized Error: 0.07 | Action: UPFIRE
Step 102, Unnormalized Error: 0.23 | Action: DOWNFIRE
Step 103, Unnormalized Error: 0.71 | Action: FIRE
Step 104, Unnormalized Error: 3.03 | Action: UPLEFT
Step 105, Unnormalized Error: 11.10 | Action: UPLEFTFIRE
Step 106, Unnormalized Error: 52.63 | Action: LEFTFIRE
Step 107, Unnormalized Error: 62.24 | Action: UP
Step 108, Unnormalized Error: 58.15 | Action: DOWN
Step 109, Unnormalized Error: 61.82 | Action: UPFIRE
Step 110, Unnormalized Error: 51.90 | Action: UPLEFTFIRE
Step 111, Unnormalized Error: 101.31 | Action: LEFTFIRE
Step 112, Unnormalized Error: 160.35 | Action: DOWNLEFT
Step 113, Unnormalized Error: 374.68 | Action: DOWN
Step 114, Unnormalized Error: 412.92 | Action: RIGHT
Step 115, Unnormalized Error: 387.29 | Action: DOWNRIGHT
Step 116, Unnormalized Error: 251.75 | Action: DOWNRIGHTFIRE
Step 117, Unnormalized Error: 165.63 | Action: DOWN
Step 118, Unnormalized Error: 116.96 | Action: RIGHT
Step 119, Unnormalized Error: 113.83 | Action: UPRIGHTFIRE
State reset
Step 120, Unnormalized Error: 29.48 | Action: UPLEFT
Step 121, Unnormalized Error: 12.35 | Action: UP
Step 122, Unnormalized Error: 5.41 | Action: DOWNRIGHT
Step 123, Unnormalized Error: 3.01 | Action: UP
Step 124, Unnormalized Error: 1.80 | Action: UP
Step 125, Unnormalized Error: 1.40 | Action: UPRIGHT
Step 126, Unnormalized Error: 1.03 | Action: DOWN
Step 127, Unnormalized Error: 0.88 | Action: UPFIRE
Step 128, Unnormalized Error: 0.80 | Action: UP
Step 129, Unnormalized Error: 1.24 | Action: DOWNRIGHT
Step 130, Unnormalized Error: 2.51 | Action: DOWN
Step 131, Unnormalized Error: 3.56 | Action: DOWN
Step 132, Unnormalized Error: 3.94 | Action: UPRIGHT
Step 133, Unnormalized Error: 4.66 | Action: LEFT
Step 134, Unnormalized Error: 5.40 | Action: DOWNRIGHTFIRE
Step 135, Unnormalized Error: 5.03 | Action: DOWNFIRE
Step 136, Unnormalized Error: 4.83 | Action: DOWN
Step 137, Unnormalized Error: 3.93 | Action: DOWN
Step 138, Unnormalized Error: 7.32 | Action: DOWNRIGHT
Step 139, Unnormalized Error: 7.75 | Action: UPLEFTFIRE
State reset
Step 140, Unnormalized Error: 1.25 | Action: DOWN
Step 141, Unnormalized Error: 0.26 | Action: DOWNRIGHT
Step 142, Unnormalized Error: 0.63 | Action: DOWNRIGHT
Step 143, Unnormalized Error: 0.80 | Action: DOWNLEFTFIRE
Step 144, Unnormalized Error: 1.07 | Action: UPLEFTFIRE
Step 145, Unnormalized Error: 0.82 | Action: DOWN
Step 146, Unnormalized Error: 0.74 | Action: UPLEFT
Step 147, Unnormalized Error: 0.48 | Action: LEFTFIRE
Step 148, Unnormalized Error: 0.47 | Action: UPFIRE
Step 149, Unnormalized Error: 0.46 | Action: DOWNLEFTFIRE
Step 150, Unnormalized Error: 0.44 | Action: UPLEFT
Step 151, Unnormalized Error: 0.54 | Action: DOWN
Step 152, Unnormalized Error: 0.58 | Action: DOWNRIGHT
Step 153, Unnormalized Error: 0.51 | Action: DOWNFIRE
Step 154, Unnormalized Error: 1.31 | Action: UPLEFT
Step 155, Unnormalized Error: 4.75 | Action: DOWN
Step 156, Unnormalized Error: 3.63 | Action: RIGHTFIRE
Step 157, Unnormalized Error: 52.11 | Action: LEFT
Step 158, Unnormalized Error: 4.94 | Action: DOWNLEFTFIRE
Step 159, Unnormalized Error: 10.09 | Action: NOOP
State reset
Step 160, Unnormalized Error: 0.18 | Action: FIRE
Step 161, Unnormalized Error: 0.51 | Action: LEFTFIRE
Step 162, Unnormalized Error: 0.24 | Action: DOWNLEFT
Step 163, Unnormalized Error: 0.13 | Action: DOWN
Step 164, Unnormalized Error: 0.55 | Action: UPFIRE
Step 165, Unnormalized Error: 0.09 | Action: DOWNRIGHTFIRE
Step 166, Unnormalized Error: 0.17 | Action: UPLEFTFIRE
Step 167, Unnormalized Error: 0.27 | Action: FIRE
Step 168, Unnormalized Error: 0.67 | Action: LEFT
Step 169, Unnormalized Error: 0.28 | Action: UPLEFT
Step 170, Unnormalized Error: 2.28 | Action: FIRE
Step 171, Unnormalized Error: 2.05 | Action: UPLEFTFIRE
Step 172, Unnormalized Error: 0.45 | Action: DOWN
Step 173, Unnormalized Error: 4.34 | Action: RIGHTFIRE
Step 174, Unnormalized Error: 9.30 | Action: DOWN
Step 175, Unnormalized Error: 12.18 | Action: LEFTFIRE
Step 176, Unnormalized Error: 17.65 | Action: UPLEFTFIRE
Step 177, Unnormalized Error: 244.85 | Action: UPLEFT
Step 178, Unnormalized Error: 267.28 | Action: DOWN
Step 179, Unnormalized Error: 227.29 | Action: RIGHT
State reset
Step 180, Unnormalized Error: 184.72 | Action: LEFT
Step 181, Unnormalized Error: 17.77 | Action: UPFIRE
Step 182, Unnormalized Error: 17.89 | Action: UPRIGHTFIRE
Step 183, Unnormalized Error: 28.47 | Action: DOWN
Step 184, Unnormalized Error: 16.22 | Action: UP
Step 185, Unnormalized Error: 14.20 | Action: DOWN
Step 186, Unnormalized Error: 13.48 | Action: UPLEFT
Step 187, Unnormalized Error: 15.25 | Action: UPFIRE
Step 188, Unnormalized Error: 23.27 | Action: DOWN
Step 189, Unnormalized Error: 6.47 | Action: DOWN
Step 190, Unnormalized Error: 7.57 | Action: UPLEFT
Step 191, Unnormalized Error: 157.94 | Action: LEFTFIRE
Step 192, Unnormalized Error: 181.12 | Action: NOOP
Step 193, Unnormalized Error: 245.86 | Action: RIGHT
Step 194, Unnormalized Error: 161.04 | Action: UPLEFTFIRE
Step 195, Unnormalized Error: 115.27 | Action: FIRE
Step 196, Unnormalized Error: 98.12 | Action: DOWNRIGHTFIRE
Step 197, Unnormalized Error: 87.83 | Action: DOWN
Step 198, Unnormalized Error: 93.14 | Action: DOWNRIGHTFIRE
Step 199, Unnormalized Error: 95.71 | Action: UPLEFTFIRE
State reset
Step 200, Unnormalized Error: 5.95 | Action: DOWN
Step 201, Unnormalized Error: 2.24 | Action: DOWN
Step 202, Unnormalized Error: 1.44 | Action: UPRIGHT
Step 203, Unnormalized Error: 0.91 | Action: DOWN
Step 204, Unnormalized Error: 0.77 | Action: RIGHTFIRE
Step 205, Unnormalized Error: 0.74 | Action: DOWN
Step 206, Unnormalized Error: 0.87 | Action: DOWN
Step 207, Unnormalized Error: 0.72 | Action: DOWNFIRE
Step 208, Unnormalized Error: 0.66 | Action: DOWN
Step 209, Unnormalized Error: 1.73 | Action: NOOP
Step 210, Unnormalized Error: 0.82 | Action: UPRIGHT
Step 211, Unnormalized Error: 0.64 | Action: DOWN
Step 212, Unnormalized Error: 0.57 | Action: UPRIGHT
Step 213, Unnormalized Error: 1.03 | Action: DOWNLEFTFIRE
Step 214, Unnormalized Error: 1.00 | Action: UPFIRE
Step 215, Unnormalized Error: 0.98 | Action: DOWN
Step 216, Unnormalized Error: 1.17 | Action: FIRE
Step 217, Unnormalized Error: 1.11 | Action: UPLEFT
Step 218, Unnormalized Error: 2.63 | Action: FIRE
Step 219, Unnormalized Error: 3.20 | Action: NOOP
State reset
Step 220, Unnormalized Error: 0.66 | Action: UPRIGHTFIRE
Step 221, Unnormalized Error: 0.13 | Action: DOWNRIGHT
Step 222, Unnormalized Error: 0.15 | Action: NOOP
Step 223, Unnormalized Error: 0.16 | Action: UPRIGHT
Step 224, Unnormalized Error: 0.21 | Action: RIGHT
Step 225, Unnormalized Error: 0.16 | Action: DOWNRIGHT
Step 226, Unnormalized Error: 0.28 | Action: DOWNLEFTFIRE
Step 227, Unnormalized Error: 0.45 | Action: FIRE
Step 228, Unnormalized Error: 0.56 | Action: DOWN
Step 229, Unnormalized Error: 0.80 | Action: LEFTFIRE
Step 230, Unnormalized Error: 0.94 | Action: RIGHTFIRE
Step 231, Unnormalized Error: 1.10 | Action: UPLEFTFIRE
Step 232, Unnormalized Error: 1.45 | Action: DOWNLEFTFIRE
Step 233, Unnormalized Error: 2.46 | Action: UP
Step 234, Unnormalized Error: 8.74 | Action: UPRIGHT
Step 235, Unnormalized Error: 8.09 | Action: UPLEFT
Step 236, Unnormalized Error: 5.78 | Action: UP
Step 237, Unnormalized Error: 4.02 | Action: UPRIGHTFIRE
Step 238, Unnormalized Error: 6.88 | Action: DOWN
Step 239, Unnormalized Error: 18.43 | Action: UPRIGHT
State reset
Step 240, Unnormalized Error: 0.79 | Action: DOWNFIRE
Step 241, Unnormalized Error: 0.49 | Action: DOWN
Step 242, Unnormalized Error: 0.78 | Action: UPLEFT
Step 243, Unnormalized Error: 0.66 | Action: FIRE
Step 244, Unnormalized Error: 0.59 | Action: DOWN
Step 245, Unnormalized Error: 0.31 | Action: DOWNRIGHTFIRE
Step 246, Unnormalized Error: 0.26 | Action: DOWN
Step 247, Unnormalized Error: 0.23 | Action: DOWNLEFTFIRE
Step 248, Unnormalized Error: 0.24 | Action: RIGHTFIRE
Step 249, Unnormalized Error: 0.21 | Action: UPLEFT
Step 250, Unnormalized Error: 0.32 | Action: DOWN
Step 251, Unnormalized Error: 0.40 | Action: DOWNRIGHTFIRE
Step 252, Unnormalized Error: 0.28 | Action: UPRIGHTFIRE
Step 253, Unnormalized Error: 1.03 | Action: LEFTFIRE
Step 254, Unnormalized Error: 0.28 | Action: UP
Step 255, Unnormalized Error: 0.21 | Action: DOWNFIRE
Step 256, Unnormalized Error: 0.22 | Action: LEFTFIRE
Step 257, Unnormalized Error: 0.18 | Action: NOOP
Step 258, Unnormalized Error: 0.15 | Action: DOWN
Step 259, Unnormalized Error: 0.17 | Action: DOWNFIRE
State reset
Step 260, Unnormalized Error: 0.03 | Action: DOWN
Step 261, Unnormalized Error: 0.04 | Action: UPFIRE
Step 262, Unnormalized Error: 0.03 | Action: DOWNRIGHTFIRE
Step 263, Unnormalized Error: 0.04 | Action: UPLEFT
Step 264, Unnormalized Error: 0.10 | Action: LEFTFIRE
Step 265, Unnormalized Error: 0.11 | Action: DOWNRIGHTFIRE
Step 266, Unnormalized Error: 0.31 | Action: DOWN
Step 267, Unnormalized Error: 0.42 | Action: DOWN
Step 268, Unnormalized Error: 0.14 | Action: UPRIGHTFIRE
Step 269, Unnormalized Error: 0.14 | Action: UPFIRE
Step 270, Unnormalized Error: 0.12 | Action: RIGHTFIRE
Step 271, Unnormalized Error: 0.15 | Action: DOWN
Step 272, Unnormalized Error: 0.13 | Action: RIGHT
Step 273, Unnormalized Error: 0.21 | Action: DOWN
Step 274, Unnormalized Error: 0.09 | Action: UPRIGHTFIRE
Step 275, Unnormalized Error: 0.09 | Action: FIRE
Step 276, Unnormalized Error: 0.11 | Action: RIGHTFIRE
Step 277, Unnormalized Error: 0.09 | Action: FIRE
Step 278, Unnormalized Error: 0.20 | Action: UPRIGHTFIRE
Step 279, Unnormalized Error: 0.14 | Action: UPRIGHTFIRE
State reset
Step 280, Unnormalized Error: 0.03 | Action: DOWN
Step 281, Unnormalized Error: 0.03 | Action: DOWNLEFT
Step 282, Unnormalized Error: 0.08 | Action: NOOP
Step 283, Unnormalized Error: 0.05 | Action: UPLEFTFIRE
Step 284, Unnormalized Error: 0.07 | Action: DOWN
Step 285, Unnormalized Error: 0.11 | Action: LEFT
Step 286, Unnormalized Error: 0.18 | Action: UP
Step 287, Unnormalized Error: 0.20 | Action: RIGHTFIRE
Step 288, Unnormalized Error: 0.18 | Action: DOWN
Step 289, Unnormalized Error: 0.15 | Action: FIRE
Step 290, Unnormalized Error: 0.08 | Action: UPRIGHT
Step 291, Unnormalized Error: 0.19 | Action: DOWNLEFT
Step 292, Unnormalized Error: 0.21 | Action: DOWNRIGHT
Step 293, Unnormalized Error: 0.09 | Action: DOWN
Step 294, Unnormalized Error: 0.09 | Action: UPLEFTFIRE
Step 295, Unnormalized Error: 0.14 | Action: UPLEFT
Step 296, Unnormalized Error: 0.19 | Action: DOWNFIRE
Step 297, Unnormalized Error: 1.91 | Action: NOOP
Step 298, Unnormalized Error: 0.85 | Action: NOOP
Step 299, Unnormalized Error: 0.48 | Action: FIRE
State reset
Step 300, Unnormalized Error: 0.04 | Action: LEFT
Step 301, Unnormalized Error: 0.04 | Action: UPLEFTFIRE
Step 302, Unnormalized Error: 0.06 | Action: DOWNRIGHT
Step 303, Unnormalized Error: 0.07 | Action: DOWN
Step 304, Unnormalized Error: 0.11 | Action: LEFT
Step 305, Unnormalized Error: 0.16 | Action: DOWNLEFT
Step 306, Unnormalized Error: 0.29 | Action: DOWNLEFT
Step 307, Unnormalized Error: 0.31 | Action: DOWN
Step 308, Unnormalized Error: 0.37 | Action: UPLEFT
Step 309, Unnormalized Error: 0.61 | Action: UPRIGHTFIRE
Step 310, Unnormalized Error: 0.78 | Action: DOWNLEFTFIRE
Step 311, Unnormalized Error: 1.14 | Action: DOWNLEFTFIRE
Step 312, Unnormalized Error: 0.73 | Action: DOWN
Step 313, Unnormalized Error: 0.60 | Action: DOWNFIRE
Step 314, Unnormalized Error: 0.65 | Action: UP
Step 315, Unnormalized Error: 10.36 | Action: DOWN
Step 316, Unnormalized Error: 3.27 | Action: UPRIGHTFIRE
Step 317, Unnormalized Error: 2.76 | Action: DOWN
Step 318, Unnormalized Error: 16.67 | Action: RIGHT
Step 319, Unnormalized Error: 22.68 | Action: DOWN
State reset
Step 320, Unnormalized Error: 2.40 | Action: UPRIGHTFIRE
Step 321, Unnormalized Error: 1.60 | Action: UPRIGHTFIRE
Step 322, Unnormalized Error: 1.18 | Action: LEFT
Step 323, Unnormalized Error: 2.69 | Action: LEFTFIRE
Step 324, Unnormalized Error: 7.21 | Action: DOWN
Step 325, Unnormalized Error: 4.42 | Action: DOWNRIGHT
Step 326, Unnormalized Error: 3.65 | Action: DOWNLEFTFIRE
Step 327, Unnormalized Error: 3.61 | Action: DOWNRIGHT
Step 328, Unnormalized Error: 2.96 | Action: DOWN
Step 329, Unnormalized Error: 2.55 | Action: LEFTFIRE
Step 330, Unnormalized Error: 2.37 | Action: RIGHT
Step 331, Unnormalized Error: 1.86 | Action: DOWNFIRE
Step 332, Unnormalized Error: 1.84 | Action: DOWN
Step 333, Unnormalized Error: 1.37 | Action: DOWNLEFT
Step 334, Unnormalized Error: 0.91 | Action: DOWN
Step 335, Unnormalized Error: 23.39 | Action: UP
Step 336, Unnormalized Error: 5.37 | Action: DOWNRIGHT
Step 337, Unnormalized Error: 2.53 | Action: DOWN
Step 338, Unnormalized Error: 1.87 | Action: DOWNLEFTFIRE
Step 339, Unnormalized Error: 2.04 | Action: DOWN
State reset
Step 340, Unnormalized Error: 0.23 | Action: DOWN
Step 341, Unnormalized Error: 0.12 | Action: LEFT
Step 342, Unnormalized Error: 10.96 | Action: DOWN
Step 343, Unnormalized Error: 9.18 | Action: NOOP
Step 344, Unnormalized Error: 1.09 | Action: UPLEFTFIRE
Step 345, Unnormalized Error: 1.66 | Action: DOWNRIGHT
Step 346, Unnormalized Error: 74.80 | Action: NOOP
Step 347, Unnormalized Error: 1.30 | Action: LEFTFIRE
Step 348, Unnormalized Error: 3.12 | Action: DOWN
Step 349, Unnormalized Error: 8.88 | Action: DOWNRIGHT
Step 350, Unnormalized Error: 176.58 | Action: LEFT
Step 351, Unnormalized Error: 150.02 | Action: DOWN
Step 352, Unnormalized Error: 3.23 | Action: RIGHTFIRE
Step 353, Unnormalized Error: 2.37 | Action: UPLEFTFIRE
Step 354, Unnormalized Error: 3.32 | Action: DOWN
Step 355, Unnormalized Error: 5.66 | Action: RIGHT
Step 356, Unnormalized Error: 5.65 | Action: UPFIRE
Step 357, Unnormalized Error: 211.26 | Action: UPRIGHTFIRE
Step 358, Unnormalized Error: 10.82 | Action: UPLEFTFIRE
Step 359, Unnormalized Error: 28.36 | Action: NOOP
State reset
Step 360, Unnormalized Error: 1.85 | Action: DOWN
Step 361, Unnormalized Error: 2.33 | Action: DOWN
Step 362, Unnormalized Error: 53.87 | Action: DOWN
Step 363, Unnormalized Error: 21.34 | Action: LEFTFIRE
Step 364, Unnormalized Error: 33.10 | Action: UPRIGHT
Step 365, Unnormalized Error: 166.22 | Action: DOWNRIGHT
Step 366, Unnormalized Error: 7.71 | Action: DOWNLEFTFIRE
Step 367, Unnormalized Error: 6.55 | Action: DOWN
Step 368, Unnormalized Error: 7.39 | Action: DOWN
Step 369, Unnormalized Error: 11.51 | Action: UPFIRE
Step 370, Unnormalized Error: 7.53 | Action: DOWNRIGHT
Step 371, Unnormalized Error: 50.16 | Action: DOWN
Step 372, Unnormalized Error: 18.01 | Action: LEFTFIRE
Step 373, Unnormalized Error: 24.95 | Action: DOWNLEFT
Step 374, Unnormalized Error: 60.64 | Action: RIGHTFIRE
Step 375, Unnormalized Error: 44.62 | Action: DOWN
Step 376, Unnormalized Error: 33.32 | Action: DOWN
Step 377, Unnormalized Error: 240.11 | Action: DOWN
Step 378, Unnormalized Error: 194.45 | Action: UP
Step 379, Unnormalized Error: 15.37 | Action: UPRIGHTFIRE
State reset
Step 380, Unnormalized Error: 0.95 | Action: DOWNRIGHT
Step 381, Unnormalized Error: 0.78 | Action: UPLEFTFIRE
Step 382, Unnormalized Error: 1.08 | Action: DOWN
Step 383, Unnormalized Error: 0.50 | Action: DOWNLEFTFIRE
Step 384, Unnormalized Error: 0.94 | Action: UP
Step 385, Unnormalized Error: 2.50 | Action: LEFT
Step 386, Unnormalized Error: 0.39 | Action: LEFTFIRE
Step 387, Unnormalized Error: 0.66 | Action: UPRIGHTFIRE
Step 388, Unnormalized Error: 1.26 | Action: DOWN
Step 389, Unnormalized Error: 1.03 | Action: UPRIGHTFIRE
Step 390, Unnormalized Error: 1.07 | Action: LEFT
Step 391, Unnormalized Error: 1.44 | Action: DOWNLEFT
Step 392, Unnormalized Error: 3.47 | Action: UPRIGHTFIRE
Step 393, Unnormalized Error: 15.08 | Action: NOOP
Step 394, Unnormalized Error: 7.91 | Action: DOWN
Step 395, Unnormalized Error: 6.05 | Action: DOWNRIGHT
Step 396, Unnormalized Error: 5.85 | Action: DOWN
Step 397, Unnormalized Error: 6.33 | Action: UPFIRE
Step 398, Unnormalized Error: 12.39 | Action: DOWNRIGHTFIRE
Step 399, Unnormalized Error: 11.81 | Action: UP
State reset
Step 400, Unnormalized Error: 0.21 | Action: UPRIGHTFIRE
Step 401, Unnormalized Error: 0.09 | Action: RIGHT
Step 402, Unnormalized Error: 3.36 | Action: UP
Step 403, Unnormalized Error: 3.22 | Action: LEFT
Step 404, Unnormalized Error: 1.77 | Action: DOWN
Step 405, Unnormalized Error: 1.40 | Action: RIGHT
Step 406, Unnormalized Error: 0.51 | Action: DOWNLEFTFIRE
Step 407, Unnormalized Error: 0.60 | Action: DOWNLEFTFIRE
Step 408, Unnormalized Error: 0.79 | Action: UPLEFT
Step 409, Unnormalized Error: 0.95 | Action: DOWN
Step 410, Unnormalized Error: 1.79 | Action: LEFT
Step 411, Unnormalized Error: 1.92 | Action: DOWN
Step 412, Unnormalized Error: 2.14 | Action: DOWNLEFTFIRE
Step 413, Unnormalized Error: 2.09 | Action: LEFT
Step 414, Unnormalized Error: 3.20 | Action: FIRE
Step 415, Unnormalized Error: 1.70 | Action: UPLEFT
Step 416, Unnormalized Error: 1.72 | Action: DOWNFIRE
Step 417, Unnormalized Error: 2.17 | Action: UPLEFTFIRE
Step 418, Unnormalized Error: 2.27 | Action: DOWNLEFT
Step 419, Unnormalized Error: 1.87 | Action: DOWNLEFTFIRE
State reset
Step 420, Unnormalized Error: 0.18 | Action: LEFTFIRE
Step 421, Unnormalized Error: 0.09 | Action: DOWN
Step 422, Unnormalized Error: 0.11 | Action: LEFTFIRE
Step 423, Unnormalized Error: 0.37 | Action: DOWN
Step 424, Unnormalized Error: 0.18 | Action: UPFIRE
Step 425, Unnormalized Error: 0.22 | Action: DOWN
Step 426, Unnormalized Error: 0.15 | Action: LEFTFIRE
Step 427, Unnormalized Error: 0.22 | Action: RIGHTFIRE
Step 428, Unnormalized Error: 0.19 | Action: FIRE
Step 429, Unnormalized Error: 0.20 | Action: DOWN
Step 430, Unnormalized Error: 0.18 | Action: RIGHTFIRE
Step 431, Unnormalized Error: 0.18 | Action: NOOP
Step 432, Unnormalized Error: 0.15 | Action: DOWNLEFTFIRE
Step 433, Unnormalized Error: 0.29 | Action: LEFT
Step 434, Unnormalized Error: 0.22 | Action: DOWNLEFT
Step 435, Unnormalized Error: 0.15 | Action: DOWNLEFTFIRE
Step 436, Unnormalized Error: 0.14 | Action: UPRIGHT
Step 437, Unnormalized Error: 0.23 | Action: NOOP
Step 438, Unnormalized Error: 0.66 | Action: LEFT
Step 439, Unnormalized Error: 0.65 | Action: RIGHTFIRE
State reset
Step 440, Unnormalized Error: 0.09 | Action: RIGHTFIRE
Step 441, Unnormalized Error: 0.06 | Action: UP
Step 442, Unnormalized Error: 0.08 | Action: DOWN
Step 443, Unnormalized Error: 0.17 | Action: LEFTFIRE
Step 444, Unnormalized Error: 0.13 | Action: LEFTFIRE
Step 445, Unnormalized Error: 0.16 | Action: LEFT
Step 446, Unnormalized Error: 0.28 | Action: RIGHTFIRE
Step 447, Unnormalized Error: 0.26 | Action: UPRIGHTFIRE
Step 448, Unnormalized Error: 0.30 | Action: DOWNLEFTFIRE
Step 449, Unnormalized Error: 0.37 | Action: UPFIRE
Step 450, Unnormalized Error: 0.42 | Action: FIRE
Step 451, Unnormalized Error: 0.33 | Action: DOWNRIGHTFIRE
Step 452, Unnormalized Error: 0.22 | Action: DOWNRIGHT
Step 453, Unnormalized Error: 0.58 | Action: DOWN
Step 454, Unnormalized Error: 0.23 | Action: DOWNFIRE
Step 455, Unnormalized Error: 0.25 | Action: RIGHT
Step 456, Unnormalized Error: 0.15 | Action: LEFTFIRE
Step 457, Unnormalized Error: 0.15 | Action: UPRIGHT
Step 458, Unnormalized Error: 0.17 | Action: DOWNLEFT
Step 459, Unnormalized Error: 0.79 | Action: RIGHT
State reset
Step 460, Unnormalized Error: 0.27 | Action: DOWNFIRE
Step 461, Unnormalized Error: 0.18 | Action: DOWNLEFTFIRE
Step 462, Unnormalized Error: 0.13 | Action: UPLEFT
Step 463, Unnormalized Error: 0.37 | Action: DOWN
Step 464, Unnormalized Error: 0.95 | Action: DOWN
Step 465, Unnormalized Error: 0.73 | Action: UP
Step 466, Unnormalized Error: 0.51 | Action: DOWNRIGHT
Step 467, Unnormalized Error: 0.41 | Action: UPRIGHTFIRE
Step 468, Unnormalized Error: 0.37 | Action: DOWN
Step 469, Unnormalized Error: 0.73 | Action: UPLEFT
Step 470, Unnormalized Error: 0.70 | Action: FIRE
Step 471, Unnormalized Error: 0.81 | Action: DOWN
Step 472, Unnormalized Error: 0.66 | Action: DOWNRIGHT
Step 473, Unnormalized Error: 0.51 | Action: UP
Step 474, Unnormalized Error: 0.50 | Action: NOOP
Step 475, Unnormalized Error: 0.49 | Action: NOOP
Step 476, Unnormalized Error: 0.53 | Action: DOWN
Step 477, Unnormalized Error: 0.68 | Action: UPRIGHT
Step 478, Unnormalized Error: 0.64 | Action: UPFIRE
Step 479, Unnormalized Error: 0.66 | Action: DOWN
State reset
Step 480, Unnormalized Error: 0.04 | Action: DOWNFIRE
Step 481, Unnormalized Error: 0.04 | Action: RIGHT
Step 482, Unnormalized Error: 0.04 | Action: DOWNRIGHT
Step 483, Unnormalized Error: 0.04 | Action: UPRIGHT
Step 484, Unnormalized Error: 0.05 | Action: DOWN
Step 485, Unnormalized Error: 0.26 | Action: DOWN
Step 486, Unnormalized Error: 0.27 | Action: RIGHTFIRE
Step 487, Unnormalized Error: 0.30 | Action: DOWN
Step 488, Unnormalized Error: 0.49 | Action: DOWN
Step 489, Unnormalized Error: 1.06 | Action: DOWNRIGHT
Step 490, Unnormalized Error: 1.72 | Action: RIGHTFIRE
Step 491, Unnormalized Error: 1.63 | Action: DOWNRIGHT
Step 492, Unnormalized Error: 2.01 | Action: DOWNRIGHTFIRE
Step 493, Unnormalized Error: 4.26 | Action: UPFIRE
Step 494, Unnormalized Error: 9.92 | Action: FIRE
Step 495, Unnormalized Error: 8.72 | Action: LEFT
Step 496, Unnormalized Error: 9.23 | Action: NOOP
Step 497, Unnormalized Error: 8.53 | Action: DOWN
Step 498, Unnormalized Error: 5.36 | Action: UPRIGHTFIRE
Step 499, Unnormalized Error: 4.61 | Action: NOOP
Comparison completed
MBRLfhilprec@Florian:~/MBRL$ 
















# with clipping
State reset
Step 70, Unnormalized Error: 1.22 | Action: UPLEFTFIRE
Step 70: LSTM state explosion - Layer1 h:2.10 c:4.07, Layer2 h:4.51 c:8.90
Step 71, Unnormalized Error: 0.87 | Action: LEFT
Step 71: LSTM state explosion - Layer1 h:2.31 c:4.71, Layer2 h:5.82 c:12.43
Step 72, Unnormalized Error: 0.55 | Action: RIGHTFIRE
Step 72: LSTM state explosion - Layer1 h:2.31 c:4.90, Layer2 h:6.21 c:14.03
Step 73, Unnormalized Error: 0.58 | Action: DOWN
Step 73: LSTM state explosion - Layer1 h:2.27 c:4.87, Layer2 h:6.25 c:14.71
Step 74, Unnormalized Error: 0.77 | Action: UPFIRE
Step 74: LSTM state explosion - Layer1 h:2.56 c:5.34, Layer2 h:6.28 c:15.27
Step 75, Unnormalized Error: 1.07 | Action: LEFTFIRE
Step 75: LSTM state explosion - Layer1 h:2.27 c:4.74, Layer2 h:6.33 c:15.88
Step 76, Unnormalized Error: 0.79 | Action: UP
Step 76: LSTM state explosion - Layer1 h:2.29 c:4.74, Layer2 h:6.36 c:16.75
Step 77, Unnormalized Error: 1.93 | Action: UPLEFT
Step 77: LSTM state explosion - Layer1 h:2.07 c:4.49, Layer2 h:6.42 c:17.12
Step 78, Unnormalized Error: 0.58 | Action: DOWN
Step 78: LSTM state explosion - Layer1 h:2.24 c:4.77, Layer2 h:6.20 c:17.23
Step 79, Unnormalized Error: 0.45 | Action: UPRIGHT
Step 79: LSTM state explosion - Layer1 h:2.50 c:5.28, Layer2 h:6.20 c:17.53
State reset
Step 80, Unnormalized Error: 0.04 | Action: DOWN
Step 80: LSTM state explosion - Layer1 h:2.40 c:5.16, Layer2 h:6.14 c:17.81
Step 81, Unnormalized Error: 0.10 | Action: UPRIGHTFIRE
Step 81: LSTM state explosion - Layer1 h:2.64 c:5.50, Layer2 h:6.36 c:18.15
Step 82, Unnormalized Error: 0.18 | Action: FIRE
Step 82: LSTM state explosion - Layer1 h:2.51 c:5.03, Layer2 h:6.35 c:18.61
Step 83, Unnormalized Error: 0.18 | Action: DOWN
Step 83: LSTM state explosion - Layer1 h:2.05 c:4.42, Layer2 h:6.33 c:18.75
Step 84, Unnormalized Error: 0.15 | Action: DOWNRIGHTFIRE
Step 84: LSTM state explosion - Layer1 h:2.03 c:4.40, Layer2 h:6.49 c:19.43
Step 85, Unnormalized Error: 1.12 | Action: RIGHTFIRE
Step 85: LSTM state explosion - Layer1 h:1.92 c:4.26, Layer2 h:6.44 c:19.84
Step 86, Unnormalized Error: 1.52 | Action: DOWN
Step 86: LSTM state explosion - Layer1 h:2.17 c:4.61, Layer2 h:6.34 c:20.02
Step 87, Unnormalized Error: 1.01 | Action: UPRIGHT
Step 87: LSTM state explosion - Layer1 h:2.29 c:4.70, Layer2 h:6.40 c:20.38
Step 88, Unnormalized Error: 78.84 | Action: LEFTFIRE
Step 88: LSTM state explosion - Layer1 h:1.85 c:4.16, Layer2 h:6.41 c:20.09
Step 89, Unnormalized Error: 12.82 | Action: UPLEFTFIRE
Step 89: LSTM state explosion - Layer1 h:2.37 c:4.84, Layer2 h:6.37 c:20.61
State reset
Step 90, Unnormalized Error: 1.48 | Action: DOWN
Step 90: LSTM state explosion - Layer1 h:2.32 c:4.86, Layer2 h:6.28 c:20.51
Step 91, Unnormalized Error: 3.55 | Action: RIGHTFIRE
Step 91: LSTM state explosion - Layer1 h:2.39 c:5.04, Layer2 h:6.36 c:20.02
Step 92, Unnormalized Error: 3.61 | Action: UPLEFTFIRE
Step 92: LSTM state explosion - Layer1 h:2.60 c:5.34, Layer2 h:6.30 c:20.28
Step 93, Unnormalized Error: 31.40 | Action: DOWN
Step 93: LSTM state explosion - Layer1 h:2.29 c:4.94, Layer2 h:6.34 c:19.85
Step 94, Unnormalized Error: 4.36 | Action: UPLEFTFIRE
Step 94: LSTM state explosion - Layer1 h:2.43 c:5.15, Layer2 h:6.43 c:19.75
Step 95, Unnormalized Error: 6.85 | Action: DOWN
Step 95: LSTM state explosion - Layer1 h:2.35 c:5.04, Layer2 h:6.30 c:19.64
Step 96, Unnormalized Error: 8.99 | Action: LEFT
Step 96: LSTM state explosion - Layer1 h:2.31 c:5.04, Layer2 h:6.37 c:19.23
Step 97, Unnormalized Error: 15.25 | Action: NOOP
Step 97: LSTM state explosion - Layer1 h:2.53 c:5.20, Layer2 h:6.45 c:19.44
Step 98, Unnormalized Error: 6.92 | Action: FIRE
Step 98: LSTM state explosion - Layer1 h:2.48 c:5.08, Layer2 h:6.39 c:19.65
Step 99, Unnormalized Error: 3.90 | Action: DOWNFIRE
Step 99: LSTM state explosion - Layer1 h:2.31 c:4.99, Layer2 h:6.36 c:19.59
State reset
Step 100, Unnormalized Error: 0.33 | Action: DOWNRIGHT
Step 100: LSTM state explosion - Layer1 h:2.40 c:5.10, Layer2 h:6.31 c:19.90
Step 101, Unnormalized Error: 0.99 | Action: UPFIRE
Step 101: LSTM state explosion - Layer1 h:2.28 c:4.95, Layer2 h:6.48 c:19.68
Step 102, Unnormalized Error: 12.99 | Action: DOWNFIRE
Step 102: LSTM state explosion - Layer1 h:2.22 c:4.91, Layer2 h:6.44 c:19.32
Step 103, Unnormalized Error: 38.67 | Action: FIRE
Step 103: LSTM state explosion - Layer1 h:2.34 c:5.00, Layer2 h:6.36 c:19.48
Step 104, Unnormalized Error: 85.09 | Action: UPLEFT
Step 104: LSTM state explosion - Layer1 h:2.27 c:4.89, Layer2 h:6.42 c:19.19
Step 105, Unnormalized Error: 65.74 | Action: UPLEFTFIRE
Step 105: LSTM state explosion - Layer1 h:2.20 c:4.81, Layer2 h:6.54 c:18.92
Step 106, Unnormalized Error: 69.64 | Action: LEFTFIRE
Step 106: LSTM state explosion - Layer1 h:2.10 c:4.57, Layer2 h:6.38 c:18.47
Step 107, Unnormalized Error: 60.79 | Action: UP
Step 107: LSTM state explosion - Layer1 h:2.06 c:4.53, Layer2 h:6.48 c:18.19
Step 108, Unnormalized Error: 59.32 | Action: DOWN
Step 108: LSTM state explosion - Layer1 h:2.13 c:4.59, Layer2 h:6.53 c:18.82
Step 109, Unnormalized Error: 65.42 | Action: UPFIRE
Step 109: LSTM state explosion - Layer1 h:2.15 c:4.59, Layer2 h:6.61 c:19.23
State reset
Step 110, Unnormalized Error: 1.72 | Action: UPLEFTFIRE
Step 110: LSTM state explosion - Layer1 h:1.86 c:3.88, Layer2 h:6.56 c:18.28
Step 111, Unnormalized Error: 1.88 | Action: LEFTFIRE
Step 111: LSTM state explosion - Layer1 h:2.27 c:4.37, Layer2 h:6.01 c:17.61
Step 112, Unnormalized Error: 2.08 | Action: DOWNLEFT
Step 112: LSTM state explosion - Layer1 h:2.02 c:4.04, Layer2 h:5.97 c:18.16
Step 113, Unnormalized Error: 104.58 | Action: DOWN
Step 113: LSTM state explosion - Layer1 h:1.41 c:3.14, Layer2 h:6.73 c:17.58
Step 114, Unnormalized Error: 71.94 | Action: RIGHT
Step 114: LSTM state explosion - Layer1 h:1.79 c:3.83, Layer2 h:6.57 c:16.70
Step 115, Unnormalized Error: 71.23 | Action: DOWNRIGHT
Step 115: LSTM state explosion - Layer1 h:1.99 c:4.19, Layer2 h:6.38 c:16.65
Step 116, Unnormalized Error: 26.31 | Action: DOWNRIGHTFIRE
Step 116: LSTM state explosion - Layer1 h:2.26 c:4.53, Layer2 h:6.10 c:16.07
Step 117, Unnormalized Error: 22.36 | Action: DOWN
Step 117: LSTM state explosion - Layer1 h:2.05 c:4.19, Layer2 h:6.14 c:15.97
Step 118, Unnormalized Error: 21.82 | Action: RIGHT
Step 118: LSTM state explosion - Layer1 h:2.01 c:4.20, Layer2 h:6.21 c:16.11
Step 119, Unnormalized Error: 22.56 | Action: UPRIGHTFIRE
Step 119: LSTM state explosion - Layer1 h:1.81 c:3.86, Layer2 h:6.33 c:15.42
State reset
Step 120, Unnormalized Error: 1.47 | Action: UPLEFT
Step 120: LSTM state explosion - Layer1 h:1.77 c:3.88, Layer2 h:6.28 c:14.82

#without clipping
