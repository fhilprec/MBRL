# Writing Notes for Master Thesis - Chapter 3

## Date: 2025-12-04

## What I Did Today

I completed Chapter 3 (Methodology) of the thesis, which describes the technical implementation of the object-centric model-based reinforcement learning approach for Pong. The chapter includes:

### 3.1 World Model Architecture
- **LSTM-Based World Model (PongMLPDeep)**: Detailed explanation of the two-layer LSTM architecture with 128 units per layer, residual connections (factor 0.2), and layer normalization
- **Alternative Architectures Explored**: Discussed why simple MLPs, deep MLPs without LSTM, GRU-based models, and transformers were not chosen
- **State Normalization and Stability**: Explained Z-score normalization, feature-specific loss weighting (10x for ball position, 5x for velocity, 2x for paddle), gradient clipping, and other stability measures

### 3.2 Actor-Critic Integration
- **DreamerV2-Style Actor-Critic**: Described how our approach differs from standard DreamerV2 by using object-centric states directly instead of learned latent states
- **Policy Learning in Imagined Rollouts**: Detailed the two-phase process of rollout generation (30k initial states, H=7 steps) and policy optimization
- **Lambda-Return Computation**: Provided the recursive formula and backward-pass algorithm for computing λ-returns with λ=0.95

### 3.3 Training Pipeline
- **World Model Training and Experience Collection**: Described initial data collection with ball-tracking policy (160k steps), life-aware batching, and periodic retraining
- **Policy Optimization**: Explained the three-stage process (rollout generation, data preparation, actor-critic updates) with timing information

## Key Technical Details from Code Analysis

### From pong_agent.py (lines 1-1389):
- Actor network: 3-layer MLP, 64 units, ELU activation, ~1M parameters (lines 166-198)
- Critic network: 3-layer MLP, 64 units, ELU activation, distributional output (Normal distribution) (lines 201-239)
- Lambda-return implementation: Recursive backward computation using JAX scan (lines 41-116)
- Imagined rollout generation: Uses feedforward world model, batched processing (100 trajectories per batch), 7-step rollouts (lines 274-444)
- Training parameters at line 972-988: rollout_length=7, num_rollouts=30000, policy_epochs=10, actor_lr=8e-5, critic_lr=5e-4, lambda=0.95, entropy_scale=0.01, discount=0.95
- Evaluation: 50 episodes every 10 iterations (lines 1292-1332)
- World model retraining: Every 10 iterations in model-based mode, collects 160k new steps (lines 1334-1382)

### From model_architectures.py (lines 1-269):
- **CORRECTION**: World model is NOT LSTM-based!
- **PongMLPDeep** (lines 14-111): 4-layer feedforward MLP with residual connections
  - Hidden size: 256 × model_scale_factor (default scale_factor=1)
  - Architecture: Dense → LayerNorm → GELU → [3× (Dense → LayerNorm → GELU + residual)] → Output
  - **Frame prediction**: Predicts only NEW frame (12 features), then shifts frame stack
  - Input: 48 (state) + 6 (one-hot action) = 54 dimensions
  - Output: 12 dimensions (new frame), which is shifted to create 48-dim next state
  - Residual connections after layers 2, 3, 4
  - Uses GELU activation (not ReLU)
  - LayerNorm after every dense layer
  - **Total parameters**: ~600K (much smaller than I initially claimed!)

### From worldmodel_mlp.py (lines 1-1352):
- State representation: 48 features (12 object features × 4 frames, scores removed) (lines 445-466)
- Feature-specific loss weighting: ball position 10x, ball velocity 5x, paddle 2x (lines 823-842)
- Experience collection: Ball-tracking policy with 50% exploration, JAX vmap + scan for parallel episode collection (lines 507-684)
- Life-aware batching: Excludes transitions crossing episode/life boundaries (lines 691-747)
- Training: Adam optimizer, lr=3e-4, batch_size=512, 50 epochs, gradient clipping at 1.0 (lines 749-1055)
- Normalization: Z-score on all features, computed from valid transitions only (lines 778-782)
- **No multi-step training**: Despite code for 2-step sequences, use_multistep=False by default (line 1246)

### Model Differences from Standard DreamerV2:
1. **No learned representation**: Uses object-centric states directly instead of learning a latent encoder
2. **Simpler world model**: Deep feedforward MLP instead of RSSM (Recurrent State-Space Model with LSTM)
3. **No recurrence**: Frame stacking provides temporal context; no hidden states maintained across predictions
4. **Object-centric states**: 48-dimensional structured features vs. high-dimensional latent vectors
5. **Explicit physics**: Ball-paddle interactions are explicitly observable in state space
6. **Hand-crafted reward**: Uses improved_pong_reward function based on ball-paddle geometry instead of learned reward model
7. **Frame-based prediction**: Predicts only new 12-dim frame, shifts frame stack (not full 48-dim state prediction)

## TODO List for Remaining Chapters

### Chapter 4: Experiments and Results
- [ ] **Section 4.1: Experimental Setup**
  - [ ] Describe hardware (GPU model, training time)
  - [ ] List hyperparameters in a table
  - [ ] Describe evaluation protocol (50 episodes every 10 iterations)
  - [ ] Define metrics: mean reward, sample efficiency (timesteps to threshold), rollout MSE

- [ ] **Section 4.2: World Model Performance**
  - [ ] **Subsection 4.2.1: Prediction Accuracy Analysis**
    - [ ] Plot 1-step, 3-step, 5-step, 10-step prediction MSE over training epochs
    - [ ] Compare ball position MSE vs. paddle position MSE
    - [ ] Show feature-wise prediction errors
  - [ ] **Subsection 4.2.2: Long-Term Rollout Quality**
    - [ ] Generate figure showing real vs. predicted trajectories for 10-20 steps
    - [ ] Quantify error accumulation over rollout length
    - [ ] Compare LSTM vs. MLP rollout stability
  - [ ] **Subsection 4.2.3: Model Stability Assessment**
    - [ ] Plot training loss curves for world model
    - [ ] Analyze LSTM hidden state norms during rollouts
    - [ ] Discuss convergence behavior

- [ ] **Section 4.3: Policy Learning Results**
  - [ ] **Subsection 4.3.1: Sample Efficiency Comparison**
    - [ ] Plot learning curves: reward vs. environment timesteps
    - [ ] Compare model-based (imagined rollouts) vs. real environment training
    - [ ] Calculate timesteps to reach reward thresholds (-10, 0, +10, +15)
  - [ ] **Subsection 4.3.2: Final Performance Evaluation**
    - [ ] Report best mean reward ± std over 50 episodes
    - [ ] Show performance distribution (histogram or box plot)
    - [ ] Compare against baselines (random policy, ball-tracking heuristic)
  - [ ] **Subsection 4.3.3: Learning Curve Analysis**
    - [ ] Plot reward progression over 3120 iterations
    - [ ] Identify learning phases (exploration, improvement, convergence)
    - [ ] Analyze variance in performance

- [ ] **Section 4.4: Ablation Studies**
  - [ ] **Subsection 4.4.1: Impact of Object-Centric Representations**
    - [ ] Compare object-centric world model vs. pixel-based baseline
    - [ ] Measure prediction error and policy performance
    - [ ] **REQUEST FIGURE**: Side-by-side visualization of object-centric state predictions vs. pixel predictions
  - [ ] **Subsection 4.4.2: Architecture Component Analysis**
    - [ ] Compare LSTM vs. MLP world models
    - [ ] Test different LSTM sizes (64, 128, 256 units)
    - [ ] Ablate residual connections (with/without, different scaling factors)
    - [ ] Ablate layer normalization
    - [ ] **REQUEST FIGURE**: Architecture comparison chart showing prediction accuracy vs. model size
  - [ ] **Subsection 4.4.3: Reward Function Design Effects**
    - [ ] Compare improved_pong_reward vs. sparse score-based reward
    - [ ] Test different reward weightings for ball-paddle interactions
    - [ ] Analyze learning speed with different reward functions
  - [ ] **Subsection 4.4.4: Real vs. Model Comparison**
    - [ ] Compare policy trained on imagined rollouts vs. real environment rollouts
    - [ ] Quantify model exploitation (policies that work in model but not reality)
    - [ ] Show benefit of periodic world model retraining
  - [ ] **Subsection 4.4.5: Policy Behavior Visualization**
    - [ ] Generate action probability heatmaps over state space
    - [ ] Show attention to ball position (saliency maps)
    - [ ] Compare learned policy to heuristic ball-tracking policy
    - [ ] **REQUEST FIGURE**: Trajectory visualization showing agent's decision-making

### Chapter 5: Discussion
- [ ] **Section 5.1: Analysis of Results**
  - [ ] Summarize key findings
  - [ ] Compare to research questions from Chapter 1
  - [ ] Discuss strengths: sample efficiency, interpretability, structured representations

- [ ] **Section 5.2: Limitations and Challenges**
  - [ ] World model prediction errors compound over long rollouts
  - [ ] Hand-crafted reward function limits generalization
  - [ ] LSTM-based model slower than MLP alternatives
  - [ ] Limited to single-object interaction (Pong)

- [ ] **Section 5.3: Comparison with Existing Methods**
  - [ ] Compare to DreamerV2 (latent space vs. object-centric)
  - [ ] Compare to model-free methods (PPO, DQN on Pong)
  - [ ] Position in OCRL literature (OCAtari benchmarks)

- [ ] **Section 5.4: Implications for Object-Centric RL**
  - [ ] Benefits of structured representations for RL
  - [ ] Trade-offs between object discovery and provided states
  - [ ] Scalability to multi-object environments

- [ ] **Section 5.5: Future Research Directions**
  - [ ] Extend to more complex Atari games (Breakout, Space Invaders)
  - [ ] Learn object representations from pixels (combine with slot attention)
  - [ ] Multi-step world model training (2-step, 4-step consistency)
  - [ ] Causal reasoning over object interactions
  - [ ] Transfer learning across games with shared object types

### Chapter 6: Conclusion
- [ ] Summarize contributions
- [ ] Answer research questions directly
- [ ] Highlight key achievements: X reward in Y timesteps
- [ ] Final thoughts on object-centric MBRL

### Appendix
- [ ] **Appendix A: Implementation Details**
  - [ ] Code structure diagram
  - [ ] Full hyperparameter table
  - [ ] JAX/Flax implementation notes
- [ ] **Appendix B: Additional Results**
  - [ ] Extended learning curves
  - [ ] Additional ablation studies
  - [ ] Failed experiments and lessons learned

## Figures and Tables to Generate

### High Priority (for Chapter 4):
1. **Figure 4.1**: World model prediction accuracy over training epochs (1-step, 5-step, 10-step MSE)
2. **Figure 4.2**: Real vs. predicted trajectory visualization (10-step rollout, showing ball and paddle positions)
3. **Figure 4.3**: Learning curves comparing imagined vs. real rollouts (reward vs. timesteps)
4. **Figure 4.4**: LSTM vs. MLP world model comparison (prediction MSE and policy reward)
5. **Figure 4.5**: Policy action distribution heatmap (state space → action probabilities)
6. **Table 4.1**: Hyperparameter configuration summary
7. **Table 4.2**: Final performance comparison (mean ± std reward, timesteps to convergence)

### Medium Priority (for Chapter 4 ablations):
8. **Figure 4.6**: Reward function comparison (sparse vs. dense reward, learning speed)
9. **Figure 4.7**: Architecture ablation results (bar chart of different components)
10. **Figure 4.8**: World model retraining impact (performance with/without periodic retraining)

### Nice to Have (for Appendix):
11. **Figure A.1**: Complete training pipeline flowchart
12. **Figure A.2**: World model architecture diagram (LSTM layers, residual connections)
13. **Figure A.3**: Actor-critic network architectures

## Code Observations and Notes

### Strengths:
- Clean separation between world model training (worldmodel_mlp.py) and policy training (pong_agent.py)
- Extensive use of JAX JIT compilation and vectorization (vmap, scan) for speed
- Feature-specific loss weighting shows domain knowledge
- Life-aware batching prevents learning from invalid transitions
- Residual connections in world model improve stability

### Potential Improvements to Discuss:
- Could explore multi-step world model training (currently single-step)
- Reward predictor code exists but is not actively used (could be revisited)
- MODEL_SCALE_FACTOR is set to 5 in pong_agent but 1 in worldmodel_mlp (document this!)
- Hard-coded reward function could be learned
- 7-step rollouts might be short for long-term planning

### Questions to Address in Discussion:
1. Why is the world model prediction error still significant after 50 epochs?
2. How much does model exploitation affect real environment performance?
3. Could transformer architectures work better with longer context?
4. What is the optimal rollout length vs. prediction accuracy trade-off?
5. How does this approach scale to games with more objects (Breakout: multiple bricks)?

## Writing Style Notes
- Use active voice where possible
- Technical precision: define all mathematical notation
- Include code references for reproducibility (e.g., "pong_agent.py:274-444")
- Connect back to research questions throughout results
- Provide intuition alongside mathematical formulations
- Use figures to illustrate complex concepts (especially algorithms)

## References to Add
Need to add citations for:
- [x] LSTM networks (Hochreiter & Schmidhuber 1997) - hochreiter1997long
- [x] GRU networks (Cho et al. 2014) - cho2014learning
- [x] Attention/Transformers (Vaswani et al. 2017) - vaswani2017attention
- [ ] **Layer normalization** (Ba et al. 2016) - ba2016layer ⚠️ MISSING
- [ ] **GELU activation** (Hendrycks & Gimpel 2016) - hendrycks2016gaussian ⚠️ MISSING
- [x] Adam optimizer (Kingma & Ba 2014) - kingma2014adam
- [x] Orthogonal initialization (Saxe et al. 2013) - saxe2013exact
- [x] ELU activation (Clevert et al. 2015) - clevert2015fast
- [x] Distributional RL (Bellemare et al. 2017) - bellemare2017distributional
- [x] REINFORCE (Williams 1992) - williams1992simple
- [x] RL textbook (Sutton & Barto 2018) - sutton2018reinforcement
- [ ] **ResNet/Residual connections** (He et al. 2016) - he2016deep ⚠️ MISSING
- [ ] PPO (for comparison) - already in Chapter 2
- [ ] DQN (for comparison) - already in Chapter 2

## Next Steps
1. ✅ Complete Chapter 3 writing
2. ✅ Create this writing notes file
3. Run experiments to generate data for Chapter 4
4. Create visualization scripts for figures
5. Begin writing Chapter 4 sections iteratively
6. Generate all required figures and tables
7. Proofread and refine Chapter 3 based on Chapter 4 results

## Subchapter Renaming Suggestions

I've kept most subchapter names as they were well-structured. However, here are a few suggestions:

### Current: "Alternative Architectures Explored"
**Suggested:** "Architecture Design Choices and Alternatives"
- Rationale: Emphasizes the decision-making process

### Current: "Policy Learning in Imagined Rollouts"
**Suggested:** "Rollout Generation and Policy Optimization"
- Rationale: More accurately reflects the two-phase process

### Current: "World Model Training and experience collection"
**Suggested:** "Experience Collection and World Model Training"
- Rationale: Reflects chronological order (collect data first, then train)

These are minor suggestions - the current structure is already clear and logical!

## Additional Notes
- The MODEL_SCALE_FACTOR discrepancy (5 vs 1) should be explained in the implementation details
- Consider adding a figure showing the complete training pipeline (Algorithm 3.3 could be visualized)
- The improved_pong_reward function deserves more detailed explanation (currently simplified in Chapter 3)
- Consider adding a subsection on computational efficiency (training time, memory usage)
