using Random, Printf, StaticArrays

# ==============================================================================
# MCCFR 6-Max Engine in Julia
# ==============================================================================

# --- 1. State Dictionary ---
# Custom dictionary key for exact state history tracking
struct InfoSetKey
    street::Int8
    bucket::Int8
    histories::SVector{4, UInt64}
end

mutable struct CFRNode
    actions::Vector{Int8}  
    num_actions::Int
    regret_sum::Vector{Float64}
    strategy_sum::Vector{Float64}
end

function CFRNode(actions::Vector{Int8})
    n = length(actions)
    return CFRNode(actions, n, zeros(Float64, n), zeros(Float64, n))
end

function get_current_strategy!(node::CFRNode, strategy::Vector{Float64})
    normalization_sum = 0.0
    for i in 1:node.num_actions
        strategy[i] = max(node.regret_sum[i], 0.0)
        normalization_sum += strategy[i]
    end
    
    for i in 1:node.num_actions
        if normalization_sum > 0
            strategy[i] /= normalization_sum
        else
            strategy[i] = 1.0 / node.num_actions
        end
    end
end

function update_average_strategy!(node::CFRNode, current_strategy::Vector{Float64}, reach_prob::Float64)
    for i in 1:node.num_actions
        node.strategy_sum[i] += current_strategy[i] * reach_prob
    end
end

# --- 2. State Mechanics ---
struct GameState
    pot::Float64
    stacks::SVector{6, Float64}
    board::SVector{5, Int}
    histories::SVector{4, UInt64} 
    num_raises::SVector{4, Int8}    
    current_street::Int8
    active_players::SVector{6, Bool}
    street_bets::SVector{6, Float64}
    current_turn::Int8
    players_at_street_start::Int8 
    actions_this_street::Int8       
end

function get_next_active_player(active_players::SVector{6, Bool}, from_index::Int8)
    for i in 1:5
        idx = Int8(mod1(from_index + i, 6))
        if active_players[idx]
            return idx
        end
    end
    return from_index
end

function is_round_closed(state::GameState)
    active_count = sum(state.active_players)
    if active_count == 1
        return true
    end
    
    if state.actions_this_street < state.players_at_street_start
        return false
    end

    max_bet = maximum(state.street_bets)
    for i in 1:6
        # Note: Exclude all-in players from betting round closure checks
        if state.active_players[i] && state.stacks[i] > 1e-5 && state.street_bets[i] < max_bet - 1e-5
            return false
        end
    end
    return true
end

function get_legal_actions(state::GameState)
    player = state.current_turn
    max_bet = maximum(state.street_bets)
    my_bet = state.street_bets[player]
    my_stack = state.stacks[player]
    
    call_cost = max_bet - my_bet
    actions = Tuple{Int8, Float64}[]
    
    # Auto-check for all-in players
    if my_stack < 1e-5
        push!(actions, (Int8(2), 0.0))
        return actions
    end
    
    # 1. FOLD
    if call_cost > 1e-5
        push!(actions, (Int8(1), 0.0)) 
    end
    
    # 2. CALL / CHECK
    actual_call_cost = min(call_cost, my_stack)
    push!(actions, (Int8(2), actual_call_cost)) 
    
    current_raises = state.num_raises[state.current_street]
    
    # =======================================================
    # Pruning action space: limited to 2 raises to maintain memory bounds
    # =======================================================
    if my_stack > call_cost + 1e-5 && current_raises < 2
        pot_after_call = state.pot + call_cost
        
        # Only allow Raise 50% (Option 4) and All-In (Option 6)
        cost_r50 = call_cost + (0.50 * pot_after_call)
        if cost_r50 < my_stack push!(actions, (Int8(4), cost_r50)) end
        
        push!(actions, (Int8(6), my_stack)) 
    end
    
    return actions
end

function generate_child_state(state::GameState, action_id::Int8, cost::Float64)
    player = state.current_turn
    street = state.current_street
    
    new_street_history = (state.histories[street] * 10) + UInt64(action_id)
    new_histories = setindex(state.histories, new_street_history, street)
    
    new_raises = state.num_raises
    if action_id >= 3 
        new_raises = setindex(new_raises, new_raises[street] + Int8(1), street)
    end
    
    if action_id == 1 
        new_active_players = setindex(state.active_players, false, player)
        new_stacks = state.stacks
        new_pot = state.pot
        new_bets = state.street_bets
    else
        new_active_players = state.active_players
        new_stacks = setindex(state.stacks, state.stacks[player] - cost, player)
        new_pot = state.pot + cost
        new_bets = setindex(state.street_bets, state.street_bets[player] + cost, player)
    end
    
    new_actions_this_street = state.actions_this_street + Int8(1)
    new_players_at_start = state.players_at_street_start
    
    temp_state = GameState(new_pot, new_stacks, state.board, new_histories, 
                           new_raises, street, new_active_players, new_bets, player,
                           new_players_at_start, new_actions_this_street)
    
    if is_round_closed(temp_state)
        if sum(new_active_players) == 1 || street == 4
            next_turn = player
        else
            street += Int8(1)
            new_bets = @SVector zeros(Float64, 6)
            next_turn = get_next_active_player(new_active_players, Int8(6)) 
            new_actions_this_street = Int8(0)
            new_players_at_start = Int8(sum(new_active_players))
        end
    else
        next_turn = get_next_active_player(new_active_players, player)
    end
    
    return GameState(new_pot, new_stacks, state.board, new_histories, 
                     new_raises, street, new_active_players, new_bets, next_turn,
                     new_players_at_start, new_actions_this_street)
end

function abstract_hand(player_cards::SVector{2, Int}, board_cards::SVector{5, Int}, street::Int8)
    sum_val = sum(player_cards)
    limit = street == 1 ? 0 : (street == 2 ? 3 : (street == 3 ? 4 : 5))
    for i in 1:limit
        sum_val += board_cards[i]
    end
    return Int8(mod(sum_val, 10)) 
end

function calculate_pot_distribution(state::GameState, hands::SVector{6, SVector{2, Int}}, board::SVector{5, Int})
    utilities = @MVector zeros(Float64, 6) 
    active_players = state.active_players
    
    if sum(active_players) == 1
        winner = findfirst(active_players)
        for i in 1:6
            utilities[i] = (i == winner) ? (state.pot - state.street_bets[i]) : -state.street_bets[i]
        end
        return utilities
    end
    
    best_score = -1
    winner = 0
    for i in 1:6
        if active_players[i]
            score = sum(hands[i]) + sum(board)
            if score > best_score
                best_score = score
                winner = i
            end
        end
    end
    
    for i in 1:6
        if i == winner
            utilities[i] = state.pot - state.street_bets[i]
        elseif active_players[i]
            utilities[i] = -state.street_bets[i]
        end
    end
    return utilities
end

const node_dictionary = Dict{InfoSetKey, CFRNode}()

function mccfr(state::GameState, probs::SVector{6, Float64}, hands::SVector{6, SVector{2, Int}}, board::SVector{5, Int})
    if is_round_closed(state)
        if sum(state.active_players) == 1 || state.current_street == 4
            return calculate_pot_distribution(state, hands, board)
        end
    end
    
    player = state.current_turn
    street = state.current_street
    
    bucket = abstract_hand(hands[player], board, street)
    
    # Store full history vector in the information set key
    node_key = InfoSetKey(street, bucket, state.histories)
    
    legal_actions = get_legal_actions(state)
    if !haskey(node_dictionary, node_key)
        node_dictionary[node_key] = CFRNode([a[1] for a in legal_actions])
    end
    
    node = node_dictionary[node_key]
    strategy = zeros(Float64, node.num_actions)
    get_current_strategy!(node, strategy)
    
    utilities_per_action = zeros(Float64, node.num_actions)
    expected_node_utility = @MVector zeros(Float64, 6)
    
    for (i, (action_id, cost)) in enumerate(legal_actions)
        child_state = generate_child_state(state, action_id, cost)
        
        new_probs = setindex(probs, probs[player] * strategy[i], player)
        future_utilities = mccfr(child_state, new_probs, hands, board)
        
        utilities_per_action[i] = future_utilities[player]
        
        for j in 1:6
            expected_node_utility[j] += strategy[i] * future_utilities[j]
        end
    end
    
    universe_prob = 1.0
    for j in 1:6
        if j != player
            universe_prob *= probs[j]
        end
    end
    
    for i in 1:node.num_actions
        node.regret_sum[i] += (utilities_per_action[i] - expected_node_utility[player]) * universe_prob
    end
    
    update_average_strategy!(node, strategy, probs[player])
    return expected_node_utility
end

function train(iterations::Int)
    for iter in 1:iterations
        deck = randperm(52)
        board = SVector{5, Int}(deck[1:5])
        hands = SVector{6, SVector{2, Int}}(
            SVector{2, Int}(deck[6:7]), SVector{2, Int}(deck[8:9]),
            SVector{2, Int}(deck[10:11]), SVector{2, Int}(deck[12:13]),
            SVector{2, Int}(deck[14:15]), SVector{2, Int}(deck[16:17])
        )
        
        # Scenario initialization: Limped Pot SB vs BB (100 BB)
        state = GameState(
            1.5, 
            SVector{6, Float64}(99.5, 99.0, 100.0, 100.0, 100.0, 100.0), 
            board, 
            SVector{4, UInt64}(0, 0, 0, 0), 
            SVector{4, Int8}(0, 0, 0, 0),
            Int8(1), 
            
            # Heads-up isolation: only SB and BB active
            SVector{6, Bool}(true, true, false, false, false, false), 
            
            SVector{6, Float64}(0.5, 1.0, 0.0, 0.0, 0.0, 0.0), 
            Int8(1),
            Int8(2), 
            Int8(0)  
        )
        
        probs = SVector{6, Float64}(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
        mccfr(state, probs, hands, board)
    end
end

using SQLite, JSON3

function export_to_sqlite(memory_dictionary, filename="gto_100bb.sqlite")
    println("\nStarting compression and SQLite export...")
    
    db = SQLite.DB(filename)
    SQLite.execute(db, "DROP TABLE IF EXISTS gto_nodes")
    SQLite.execute(db, "CREATE TABLE gto_nodes (situation TEXT PRIMARY KEY, strategy TEXT)")
    
    stmt = SQLite.Stmt(db, "INSERT INTO gto_nodes (situation, strategy) VALUES (?, ?)")
    action_names = ["f", "c", "r33", "r50", "r75", "rAI"]
    exported_nodes = 0
    
    SQLite.transaction(db) do
        for (key, node) in memory_dictionary
            sum_val = sum(node.strategy_sum)
            if sum_val < 1.0 continue end 
            
            final_strategy = node.strategy_sum ./ sum_val
            android_dict = Dict{String, Int}()
            
            for i in 1:node.num_actions
                int_prob = round(Int, final_strategy[i] * 100)
                if int_prob > 0
                    action_id = node.actions[i]
                    android_dict[action_names[action_id]] = int_prob
                end
            end
            
            if isempty(android_dict) continue end
            
            # Safe string formatting all 4 streets
            h = key.histories
            situation_str = "C$(key.street)_Bucket$(key.bucket)_H$(h[1])_$(h[2])_$(h[3])_$(h[4])"
            SQLite.execute(stmt, (situation_str, JSON3.write(android_dict)))
            exported_nodes += 1
        end
    end
    
    println("Success: $exported_nodes nodes have been exported to '$filename'")
end

# =========================================================
# Execution and Benchmarking
# =========================================================

println("Pre-compiling JIT engine (not measured in time)...")
train(1)

test_iterations = 500
println("\nStarting massive benchmark of $test_iterations iterations...")
println("Tree: 6-Max Full (100 big blinds)")

time_seconds = @elapsed train(test_iterations)

iterations_per_sec = test_iterations / time_seconds
total_time_sec = 100_000_000 / iterations_per_sec
total_hours = total_time_sec / 3600

println("\n==================================================")
println("BENCHMARK RESULTS:")
println("Test time: $(round(time_seconds, digits=2)) seconds")
println("Processing speed: $(round(iterations_per_sec, digits=0)) iterations/second")
println("ESTIMATED TIME FOR 100 MILLION: $(round(total_hours, digits=1)) hours")
println("==================================================")

export_to_sqlite(node_dictionary, "gto_prueba_100bb.sqlite")