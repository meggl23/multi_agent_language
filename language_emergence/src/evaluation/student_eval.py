# based on cells 13, 14, 18
import math as mt
import random
from tqdm import tqdm
import networkx as nx
import torch
import torch.nn as nn

from src.env.gridworld import SquareGridworld, state_int_to_tuple
from src.models.dqn import DQN
from src.utils.graph_utils import graph_from_walls
from src.rl.teacher import q_matrix_from_network

def train_student(label_dict, wall_state_dict, sopt_dict, message_dict, config, device,
                  num_eps=750, max_steps=50, alpha=16.0):
    
    student = DQN(K=config.K, n_actions=config.n_actions, device=device)
    optimizer = torch.optim.Adam(student.parameters(), lr=config.lr_teacher)
    
    # Kappa balances regularization vs finding the goal. 
    # Defaulting to 0.05 in case it isn't explicitly in your config.
    kappa = getattr(config, 'kappa', 0.05)

    print("Precomputing analytical task environments...")
    tasks_data = []
    
    for task, (wall_index, init_state, goal_state) in tqdm(label_dict.items(), desc="Precomputing tasks"):
        wall_states = wall_state_dict[wall_index]
        
        G = graph_from_walls(wall_states, config)
        if not nx.is_connected(G):
            continue

        env = SquareGridworld(init_state, goal_state, wall_states, config)
        outcomes = env.get_outcomes()
        
        # 1. Precompute fixed next states
        next_states_dict = {}
        for s in range(config.grid_dim**2):
            next_states = []
            for a in range(config.n_actions):
                ns, _ = outcomes[(s, a)]
                if ns is None:
                    ns = goal_state
                next_states.append(ns)
            next_states_dict[s] = next_states

        # 2. Build the analytical probas_transformer
        # This maps the flat (action*state) array into the mathematical transition matrix space
        probas_transformer = torch.zeros(
            (config.n_actions * config.grid_dim**2, config.grid_dim**2 * config.grid_dim**2),
            device=device
        )

        for s in range(config.grid_dim**2):
            if s == goal_state:
                # Goal state is absorbing: loops back to itself
                for a in range(config.n_actions):
                    probas_transformer[a * config.grid_dim**2 + s, s * config.grid_dim**2 + s] = 1
            else:
                for a, ns in enumerate(next_states_dict[s]):
                    probas_transformer[a * config.grid_dim**2 + s, s * config.grid_dim**2 + ns] = 1

        # 3. Pre-batch all state tensors for simultaneous Q-value generation
        state_tensors = []
        for s in range(config.grid_dim**2):
            st = state_int_to_tuple(s, config, device)[0] 
            state_tensors.append(st)
        all_states_tensor = torch.stack(state_tensors)

        message = message_dict[(wall_index, init_state, goal_state)]
        opt_steps = sopt_dict.get(task, max_steps)

        tasks_data.append({
            'task': task,
            'init_state': init_state,
            'goal_state': goal_state,
            'opt_steps': opt_steps,
            'message': message,
            'all_states_tensor': all_states_tensor,
            'probas_transformer': probas_transformer
        })

    print(f"Training analytical student simultaneously on {len(tasks_data)} tasks...")
    
    # num_eps effectively acts as 'epochs' over the dataset now
    for ep in tqdm(range(num_eps), desc="Training student (epochs)"):
        random.shuffle(tasks_data)
        
        optimizer.zero_grad(set_to_none=True)
        
        for t_data in tasks_data:
            # Expand the message so it matches the number of states (grid_dim**2)
            message_rep = t_data['message'].unsqueeze(0).expand(config.grid_dim**2, -1)
            
            # input_batch shape: (grid_dim**2, state_dim + K)
            input_batch = torch.cat([t_data['all_states_tensor'], message_rep], dim=1)
            
            # Get Q-values for ALL states simultaneously. Shape: (grid_dim**2, n_actions)
            Q = student(input_batch) 
            
            # Convert Q-values to action probabilities via Softmax
            action_probas = torch.nn.functional.softmax(Q, dim=1)
            
            # Transpose and flatten to match the original reference shape (action0_state0... action1_state0...)
            action_probas_t = action_probas.t() # Shape: (n_actions, grid_dim**2)
            action_probas_flat = action_probas_t.flatten().unsqueeze(0) # Shape: (1, n_actions * grid_dim**2)
            
            # Multiply by transformer to map into a flat state-to-state probability matrix
            matrix_big_flat = action_probas_flat @ t_data['probas_transformer'] 
            
            # Reshape into (s, ns) and transpose to (ns, s) transition matrix
            matrix_big = matrix_big_flat.view(config.grid_dim**2, config.grid_dim**2).t()
            
            # Find the exact probability of reaching the goal by powering the transition matrix
            opt_steps = t_data['opt_steps']
            goal_proba = torch.linalg.matrix_power(matrix_big, opt_steps)[t_data['goal_state'], t_data['init_state']]
            
            # Original Loss Formulation:
            # 1. Distance to probability 1 (finding the goal)
            # 2. Regularization (keeping Q values low for stability)
            loss = (1 - kappa) * ((1 - goal_proba) ** 4) + (kappa / mt.sqrt(config.grid_dim**2 * config.n_actions)) * torch.norm(Q, 2)
            
            # Accumulate gradients (Memory efficient batched backprop)
            loss.backward() 
            
        optimizer.step()
        
        if (ep + 1) % 50 == 0:
            print(f"Epoch {ep + 1}/{num_eps} completed")

    return student

def train_student_with_feedback(label_dict, wall_state_dict, sopt_dict, q_matrix_dict, autoencoder, student, config, device, num_eps=750, max_steps=50):

    #initialize network and optimizer
    optimizer = torch.optim.Adam(list(autoencoder.parameters()) + list(student.parameters()), lr=config.lr_teacher)
    
    kappa = getattr(config, 'kappa', 0.05)
    gamma_sparse = getattr(config, 'gamma_sparse', 0.1)
    zeta_std = getattr(config, 'zeta_std', 5.0) 

    autoencoder.train()
    student.train()

    tasks_data = []
    
    #iterate over gridworlds
    for task, (wall_index, init_state, goal_state) in tqdm(label_dict.items(), desc="Precomputing tasks"):
        wall_states = wall_state_dict[wall_index]
        
        G = graph_from_walls(wall_states, config)
        if not nx.is_connected(G):
            continue

        #initialize environment
        env = SquareGridworld(init_state, goal_state, wall_states, config)
        outcomes = env.get_outcomes()
        
        #dictionary to retreive next state and reward given current (s,a)
        next_states_dict = {}
        for s in range(config.grid_dim**2):
            next_states = []
            for a in range(config.n_actions):
                ns, _ = outcomes[(s, a)]
                if ns is None:
                    ns = goal_state
                next_states.append(ns)
            next_states_dict[s] = next_states

        probas_transformer = torch.zeros(
            (config.n_actions * config.grid_dim**2, config.grid_dim**2 * config.grid_dim**2),
            device=device
        )

        for s in range(config.grid_dim**2):
            if s == goal_state:
                for a in range(config.n_actions):
                    probas_transformer[a * config.grid_dim**2 + s, s * config.grid_dim**2 + s] = 1
            else:
                for a, ns in enumerate(next_states_dict[s]):
                    probas_transformer[a * config.grid_dim**2 + s, s * config.grid_dim**2 + ns] = 1

        state_tensors = []
        for s in range(config.grid_dim**2):
            st = state_int_to_tuple(s, config, device)[0] 
            state_tensors.append(st)
        all_states_tensor = torch.stack(state_tensors)

        tasks_data.append({
            'task': task,
            'init_state': init_state,
            'goal_state': goal_state,
            'opt_steps': sopt_dict.get(task, max_steps),
            'original_q': q_matrix_dict[task].unsqueeze(0).to(device),
            'all_states_tensor': all_states_tensor,
            'probas_transformer': probas_transformer
        })
    
    #now the actual training loop
    for ep in tqdm(range(num_eps), desc="Joint Training (Epochs)"):
        random.shuffle(tasks_data)
        
        #set to none is very important as otherwise often some small gradients remain!!
        optimizer.zero_grad(set_to_none=True)
        
        for t_data in tasks_data:
            #do a forward pass of the autoencoder, i.e. encoding and decoding
            original_q = t_data['original_q']
            message, q_recon = autoencoder(original_q)
            flat_message = message.squeeze() 
            
            #student decoding - the student network creates its q-matrix from the message(s)
            message_rep = flat_message.unsqueeze(0).expand(config.grid_dim**2, -1)
            input_batch = torch.cat([t_data['all_states_tensor'], message_rep], dim=1)
            Q = student(input_batch) 
            
            #get action probabilities as softmax of the Q-values
            action_probas = torch.nn.functional.softmax(Q, dim=1)
            action_probas_flat = action_probas.t().flatten().unsqueeze(0) 
            matrix_big_flat = action_probas_flat @ t_data['probas_transformer'] 
            
            #create a big transition matrix
            matrix_big = matrix_big_flat.view(config.grid_dim**2, config.grid_dim**2).t()
            
            #For each student step, Apply the transition matrix to the initial probability distribution to get the new probability distribution
            goal_proba = torch.linalg.matrix_power(matrix_big, t_data['opt_steps'])[t_data['goal_state'], t_data['init_state']]
            
            #first part - probability to find the goal (difference to 1)
            student_loss_p1 = (1 - kappa) * ((1 - goal_proba) ** 4)
            #second part - keep overall Q-values low to avoid some local minima and enhance overall stability
            student_loss_p2 = (kappa / mt.sqrt(config.grid_dim**2 * config.n_actions)) * torch.norm(Q, 2)
            student_loss = student_loss_p1 + student_loss_p2
            
            # Loss calculation for the autoencoder
            recon_loss = torch.norm(original_q - q_recon, 2)
            sparse_loss = torch.norm(message, 1)
            
            #overall loss is a combination
            batch_loss = (1 - gamma_sparse) * recon_loss + gamma_sparse * sparse_loss + zeta_std * student_loss
            
            #changing of the gradients according to the loss
            batch_loss.backward() 
            optimizer.step()
            
            #set to none is very important as otherwise often some small gradients remain!!
            optimizer.zero_grad(set_to_none=True)

    return autoencoder, student
    
def run_evaluations(student, label_dict, wall_state_dict, message_dict, sopt_dict, config, device):
    
    student.eval()
    
    informed_rates = []
    misinformed_rates = []
    
    with torch.no_grad():
        for task, (wall_index, init_state, goal_state) in label_dict.items():
            wall_states = wall_state_dict[wall_index]
            
            G = graph_from_walls(wall_states, config)
            if not nx.is_connected(G):
                continue
            
            env = SquareGridworld(init_state, goal_state, wall_states, config)
            outcomes = env.get_outcomes()
            next_states_dict = {s: [outcomes[(s,a)][0] for a in range(config.n_actions)] for s in range(config.grid_dim**2)}
            
            sopt = sopt_dict[task]
            max_steps = 2 * sopt
            
            correct_message = message_dict[(wall_index, init_state, goal_state)]
            wrong_task = random.choice([t for t in label_dict if t != task])
            wrong_wall, wrong_init, wrong_goal = label_dict[wrong_task]
            wrong_message = message_dict[(wrong_wall, wrong_init, wrong_goal)]
            
            for message, rates_list in [(correct_message, informed_rates), (wrong_message, misinformed_rates)]:
                
                # get action probabilities for every state
                action_probas = torch.zeros(config.n_actions, config.grid_dim, config.grid_dim)
                for s in range(config.grid_dim**2):
                    state = state_int_to_tuple(s, config, device)
                    input_tensor = torch.cat((state[0], message), 0).unsqueeze(0)
                    q_values = student(input_tensor).squeeze()
                    action_probas[:, s // config.grid_dim, s % config.grid_dim] = torch.softmax(q_values, dim=0)
                
                # build transition matrix
                matrix = torch.zeros(config.grid_dim**2, config.grid_dim**2)
                for s in range(config.grid_dim**2):
                    if s == goal_state:
                        matrix[s, s] = 1
                    else:
                        for a, ns in enumerate(next_states_dict[s]):
                            matrix[ns, s] += action_probas[a, s // config.grid_dim, s % config.grid_dim]
                
                # compute state occupancy probabilities
                probas = torch.zeros(config.grid_dim**2)
                probas[init_state] = 1
                for _ in range(max_steps):
                    probas = matrix @ probas
                
                rates_list.append(probas[goal_state].item())
    
    print(f"Informed student:    {100*sum(informed_rates)/len(informed_rates):.2f}%")
    print(f"Misinformed student: {100*sum(misinformed_rates)/len(misinformed_rates):.2f}%")
    
    return informed_rates, misinformed_rates

def extract_student_messages(student, autoencoder, label_dict, wall_state_dict, message_dict, config, device):
    '''
    For each task, extract the student's implied Q-matrix by forward-passing all states
    through the student DQN (given the task's teacher message), then re-encode that
    Q-matrix through the frozen SAE to produce a new message.

    Returns student_message_dict with the same key structure as message_dict:
        {(wall_index, init_state, goal_state): message_tensor of shape (K,)}
    '''
    student.eval()
    autoencoder.eval()
    student_message_dict = {}

    with torch.no_grad():
        for task, (wall_index, init_state, goal_state) in label_dict.items():
            wall_states = wall_state_dict[wall_index]
            message = message_dict[(wall_index, init_state, goal_state)].to(device)

            # Reuse the generic extractor from teacher.py: forward-passes all states
            # through `student` with the given message, returns (n_actions, grid_dim, grid_dim)
            q_student = q_matrix_from_network(student, message, wall_states, config, device)

            # Re-encode through the frozen SAE
            q_student_batch = q_student.unsqueeze(0).to(device)  # (1, n_actions, grid_dim, grid_dim)
            new_message, _ = autoencoder(q_student_batch)         # (1, K)
            student_message_dict[(wall_index, init_state, goal_state)] = new_message.squeeze(0).detach()

    return student_message_dict


def close_the_loop(student, autoencoder, label_dict, wall_state_dict,
                   message_dict, sopt_dict, config, device, num_eps=750):
    '''
    The telephone game: encode the student's learned policy back through the frozen SAE
    to produce degraded messages, then train a 2nd-generation student on those messages.

    Returns (student_gen2, student_message_dict) where student_message_dict holds the
    re-encoded messages that student_gen2 was trained on.
    '''
    print("--- Closing the Loop: extracting student messages ---")
    student_message_dict = extract_student_messages(
        student, autoencoder, label_dict, wall_state_dict, message_dict, config, device
    )

    print("--- Closing the Loop: training 2nd-generation student ---")
    student_gen2 = train_student(
        label_dict, wall_state_dict, sopt_dict, student_message_dict, config, device, num_eps=num_eps
    )

    return student_gen2, student_message_dict


#######################################  HELPER FUNCTIONS  ##########################################

def run_episode(student, message, init_state, outcomes, max_steps, config, device):
    state_int = init_state
    with torch.no_grad():
        for t in range(max_steps):
            state = state_int_to_tuple(state_int, config, device)
            input_tensor = torch.cat((state[0], message), 0).unsqueeze(0)
            action = student(input_tensor).argmax().item()
            next_state_int, reward = outcomes[(state_int, action)]
            if next_state_int is None:
                return 1  # reached goal
            state_int = next_state_int
    return 0  # did not reach goal

def run_random_episode(init_state,  outcomes, max_steps, config, smart=False):
    state_int = init_state
    for t in range(max_steps):
        if smart:
            # only pick actions that don't hit walls
            valid_actions = [a for a in range(config.n_actions) 
                           if outcomes[(state_int, a)][0] != state_int]
            action = random.choice(valid_actions) if valid_actions else random.choice(range(config.n_actions))
        else:
            action = random.choice(range(config.n_actions))
        next_state_int, reward = outcomes[(state_int, action)]
        if next_state_int is None:
            return 1
        state_int = next_state_int
    return 0