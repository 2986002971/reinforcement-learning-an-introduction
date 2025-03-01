use rand::{thread_rng, Rng};
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path;
use bincode::{serialize, deserialize};

const BOARD_ROWS: usize = 4;
const BOARD_COLS: usize = 4;
const WIN_COUNT: usize = 4;

#[derive(Clone, Debug)]
struct State {
    data: Vec<Vec<i8>>,
    winner: Option<i8>,
    hash_val: Option<i64>,
    end: Option<bool>,
}

impl State {
    fn new() -> Self {
        State {
            data: vec![vec![0; BOARD_COLS]; BOARD_ROWS],
            winner: None,
            hash_val: None,
            end: None,
        }
    }

    // 计算哈希值，为每个状态提供唯一标识
    fn hash(&mut self) -> i64 {
        if let Some(hash) = self.hash_val {
            return hash;
        }
        
        let mut hash_val = 0;
        for row in &self.data {
            for &cell in row {
                hash_val = hash_val * 3 + (cell as i64 + 1);
            }
        }
        self.hash_val = Some(hash_val);
        hash_val
    }

    // 检查游戏是否结束
    fn is_end(&mut self) -> bool {
        if let Some(end) = self.end {
            return end;
        }
    
        // 检查行
        for i in 0..BOARD_ROWS {
            for j in 0..(BOARD_COLS - WIN_COUNT + 1) { // 使用 WIN_COUNT 调整范围
                let symbol = self.data[i][j];
                if symbol != 0 {
                    let mut all_match = true;
                    for k in 1..WIN_COUNT {
                        if self.data[i][j+k] != symbol {
                            all_match = false;
                            break;
                        }
                    }
                    if all_match {
                        self.winner = Some(symbol);
                        self.end = Some(true);
                        return true;
                    }
                }
            }
        }
    
        // 检查列
        for j in 0..BOARD_COLS {
            for i in 0..(BOARD_ROWS - WIN_COUNT + 1) { // 使用 WIN_COUNT 调整范围
                let symbol = self.data[i][j];
                if symbol != 0 {
                    let mut all_match = true;
                    for k in 1..WIN_COUNT {
                        if self.data[i+k][j] != symbol {
                            all_match = false;
                            break;
                        }
                    }
                    if all_match {
                        self.winner = Some(symbol);
                        self.end = Some(true);
                        return true;
                    }
                }
            }
        }
    
        // 检查对角线 (从左上到右下)
        for i in 0..(BOARD_ROWS - WIN_COUNT + 1) {
            for j in 0..(BOARD_COLS - WIN_COUNT + 1) {
                let symbol = self.data[i][j];
                if symbol != 0 {
                    let mut all_match = true;
                    for k in 1..WIN_COUNT {
                        if self.data[i+k][j+k] != symbol {
                            all_match = false;
                            break;
                        }
                    }
                    if all_match {
                        self.winner = Some(symbol);
                        self.end = Some(true);
                        return true;
                    }
                }
            }
        }
    
        // 检查对角线 (从右上到左下)
        for i in 0..(BOARD_ROWS - WIN_COUNT + 1) {
            for j in (WIN_COUNT-1)..BOARD_COLS {
                let symbol = self.data[i][j];
                if symbol != 0 {
                    let mut all_match = true;
                    for k in 1..WIN_COUNT {
                        if self.data[i+k][j-k] != symbol {
                            all_match = false;
                            break;
                        }
                    }
                    if all_match {
                        self.winner = Some(symbol);
                        self.end = Some(true);
                        return true;
                    }
                }
            }
        }
    
        // 判断平局（与原来相同）
        let mut is_full = true;
        for row in &self.data {
            for &cell in row {
                if cell == 0 {
                    is_full = false;
                    break;
                }
            }
        }
        
        if is_full {
            self.winner = Some(0);
            self.end = Some(true);
            return true;
        }
    
        // 游戏继续
        self.end = Some(false);
        false
    }

    // 创建下一个状态
    fn next_state(&self, i: usize, j: usize, symbol: i8) -> State {
        let mut new_state = self.clone();
        new_state.data[i][j] = symbol;
        // 重置缓存的值
        new_state.hash_val = None;
        new_state.end = None;
        new_state.winner = None;
        new_state
    }

    // 打印棋盘状态
    fn print_state(&self) {
        println!("-------------");
        for i in 0..BOARD_ROWS {
            print!("| ");
            for j in 0..BOARD_COLS {
                let token = match self.data[i][j] {
                    1 => "*",
                    -1 => "x",
                    _ => "0",
                };
                print!("{} | ", token);
            }
            println!();
            println!("-------------");
        }
    }
}

// 为了在HashMap中使用
impl PartialEq for State {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

type AllStates = HashMap<i64, (State, bool)>;

// 递归生成所有可能的游戏状态
fn get_all_states_impl(current_state: &mut State, current_symbol: i8, all_states: &mut AllStates) {
    for i in 0..BOARD_ROWS {
        for j in 0..BOARD_COLS {
            if current_state.data[i][j] == 0 {
                let mut new_state = current_state.next_state(i, j, current_symbol);
                let new_hash = new_state.hash();
                
                if !all_states.contains_key(&new_hash) {
                    let is_end = new_state.is_end();
                    all_states.insert(new_hash, (new_state.clone(), is_end));
                    
                    if !is_end {
                        get_all_states_impl(&mut new_state, -current_symbol, all_states);
                    }
                }
            }
        }
    }
}

fn get_all_states() -> AllStates {
    let mut current_state = State::new();
    let current_symbol = 1;
    let mut all_states = HashMap::new();
    
    all_states.insert(current_state.hash(), (current_state.clone(), current_state.is_end()));
    get_all_states_impl(&mut current_state, current_symbol, &mut all_states);
    
    all_states
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct Player {
    estimations: HashMap<i64, f64>,
    step_size: f64,
    epsilon: f64,
    states: Vec<i64>,
    greedy: Vec<bool>,
    symbol: i8,
}

impl Player {
    fn new(step_size: f64, epsilon: f64) -> Self {
        Player {
            estimations: HashMap::new(),
            step_size,
            epsilon,
            states: Vec::new(),
            greedy: Vec::new(),
            symbol: 0,
        }
    }

    fn reset(&mut self) {
        self.states.clear();
        self.greedy.clear();
    }

    fn set_state(&mut self, state: &mut State) {
        self.states.push(state.hash());
        self.greedy.push(true);
    }

    fn set_symbol(&mut self, symbol: i8, all_states: &mut AllStates) {
        self.symbol = symbol;
        
        // 只有在estimations为空时（没有加载策略时）才初始化
        if self.estimations.is_empty() {
            for (&hash_val, (state, is_end)) in all_states.iter_mut() {
                if *is_end {
                    if let Some(winner) = state.winner {
                        if winner == self.symbol {
                            self.estimations.insert(hash_val, 1.0);
                        } else if winner == 0 {
                            self.estimations.insert(hash_val, 0.5);
                        } else {
                            self.estimations.insert(hash_val, 0.0);
                        }
                    }
                } else {
                    self.estimations.insert(hash_val, 0.5);
                }
            }
        }
    }

    // 更新价值估计
    fn backup(&mut self) {
        for i in (0..self.states.len() - 1).rev() {
            let state = self.states[i];
            let next_state = self.states[i + 1];
            
            if self.greedy[i] {
                let current_estimate = *self.estimations.get(&state).unwrap_or(&0.5);
                let next_estimate = *self.estimations.get(&next_state).unwrap_or(&0.5);
                let td_error = next_estimate - current_estimate;
                
                self.estimations.insert(state, current_estimate + self.step_size * td_error);
            }
        }
    }

    // 选择行动
    fn act(&mut self, state: &mut State, _all_states: &AllStates) -> (usize, usize, i8) {
        let mut next_states = Vec::new();
        let mut next_positions = Vec::new();
        
        for i in 0..BOARD_ROWS {
            for j in 0..BOARD_COLS {
                if state.data[i][j] == 0 {
                    let mut next_state = state.next_state(i, j, self.symbol);
                    let hash = next_state.hash();
                    
                    next_positions.push((i, j));
                    next_states.push(hash);
                }
            }
        }
        
        // 探索
        if thread_rng().gen::<f64>() < self.epsilon {
            let idx = thread_rng().gen_range(0..next_positions.len());
            let (i, j) = next_positions[idx];
            
            if let Some(last) = self.greedy.last_mut() {
                *last = false;
            }
            
            return (i, j, self.symbol);
        }
        
        // 利用
        let mut values = Vec::new();
        for (idx, &hash) in next_states.iter().enumerate() {
            let estimate = *self.estimations.get(&hash).unwrap_or(&0.5);
            values.push((estimate, idx));
        }
        
        // 随机打乱相同价值的动作
        values.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        
        let (_, idx) = values[0];
        let (i, j) = next_positions[idx];
        
        (i, j, self.symbol)
    }

    // 保存策略
    fn save_policy(&self) -> io::Result<()> {
        let filename = if self.symbol == 1 { "policy_first.bin" } else { "policy_second.bin" };
        let encoded = serialize(&self.estimations).unwrap();
        let mut file = File::create(filename)?;
        file.write_all(&encoded)?;
        Ok(())
    }

    // 使用自定义文件名保存策略
    fn save_policy_with_name(&self, filename: &str) -> io::Result<()> {
        let encoded = serialize(&self.estimations).unwrap();
        let mut file = File::create(filename)?;
        file.write_all(&encoded)?;
        Ok(())
    }
    
    // 使用自定义文件名加载策略
    fn load_policy_with_name(&mut self, filename: &str) -> io::Result<()> {
        if Path::new(filename).exists() {
            let mut file = File::open(filename)?;
            let mut bytes = Vec::new();
            file.read_to_end(&mut bytes)?;
            self.estimations = deserialize(&bytes)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        } else {
            println!("警告: 模型文件 {} 不存在", filename);
        }
        Ok(())
    }
}

struct HumanPlayer {
    symbol: i8,
    state: Option<State>,
}

impl HumanPlayer {
    fn new() -> Self {
        HumanPlayer {
            symbol: 0,
            state: None,
        }
    }

    fn reset(&mut self) {}

    fn set_state(&mut self, state: &State) {
        self.state = Some(state.clone());
    }

    fn set_symbol(&mut self, symbol: i8) {
        self.symbol = symbol;
    }

    fn act(&self) -> (usize, usize, i8) {
        if let Some(state) = &self.state {            
            // 显示坐标系统说明
            println!("输入落子位置坐标 (行 列)，例如: 0 0 表示左上角");
            println!("有效范围: 行 0-{}, 列 0-{}", BOARD_ROWS - 1, BOARD_COLS - 1);
            
            loop {
                let mut input = String::new();
                io::stdin().read_line(&mut input).expect("Failed to read input");
                
                // 解析输入的坐标
                let coords: Vec<&str> = input.trim().split_whitespace().collect();
                
                if coords.len() != 2 {
                    println!("请输入两个数字，以空格分隔!");
                    continue;
                }
                
                // 尝试转换为数字
                let row = match coords[0].parse::<usize>() {
                    Ok(num) => num,
                    Err(_) => {
                        println!("行坐标必须是有效数字!");
                        continue;
                    }
                };
                
                let col = match coords[1].parse::<usize>() {
                    Ok(num) => num,
                    Err(_) => {
                        println!("列坐标必须是有效数字!");
                        continue;
                    }
                };
                
                // 检查坐标范围
                if row >= BOARD_ROWS || col >= BOARD_COLS {
                    println!("坐标超出范围! 行: 0-{}, 列: 0-{}", 
                             BOARD_ROWS - 1, BOARD_COLS - 1);
                    continue;
                }
                
                // 检查位置是否已被占用
                if state.data[row][col] != 0 {
                    println!("该位置已被占用，请选择其他位置!");
                    continue;
                }
                
                return (row, col, self.symbol);
            }
        } else {
            (1, 1, self.symbol) // 默认中心位置
        }
    }
}

struct Judger<P1, P2> {
    p1: P1,
    p2: P2,
    current_state: State,
    all_states: AllStates,
}

impl<P1, P2> Judger<P1, P2>
where
    P1: PlayerBehavior,
    P2: PlayerBehavior,
{
    fn new(mut p1: P1, mut p2: P2, mut all_states: AllStates) -> Self {
        let current_state = State::new();
        let p1_symbol = 1;
        let p2_symbol = -1;
        
        p1.set_symbol(p1_symbol, Some(&mut all_states));
        p2.set_symbol(p2_symbol, Some(&mut all_states));
        
        Judger {
            p1,
            p2,
            current_state,
            all_states,
        }
    }

    fn reset(&mut self) {
        self.p1.reset();
        self.p2.reset();
        self.current_state = State::new();
    }

    fn play(&mut self, print_state: bool) -> i8 {
        self.reset();
        let mut current_state = State::new();
        
        self.p1.set_state(&mut current_state);
        self.p2.set_state(&mut current_state);
        
        if print_state {
            current_state.print_state();
        }
        
        let mut player_idx = 0;
        loop {
            let (i, j, symbol) = if player_idx % 2 == 0 {
                self.p1.act(&mut current_state, &self.all_states)
            } else {
                self.p2.act(&mut current_state, &self.all_states)
            };
            
            current_state = current_state.next_state(i, j, symbol);
            let hash = current_state.hash();
            
            if let Some((state, is_end)) = self.all_states.get(&hash) {
                current_state = state.clone();
                
                self.p1.set_state(&mut current_state);
                self.p2.set_state(&mut current_state);
                
                if print_state {
                    current_state.print_state();
                }
                
                if *is_end {
                    return current_state.winner.unwrap_or(0);
                }
            } else {
                // 处理意外情况
                panic!("State not found in all_states!");
            }
            
            player_idx += 1;
        }
    }
}

// 重新设计 Trait
trait PlayerBehavior {
    fn reset(&mut self);
    fn set_state(&mut self, state: &mut State);
    fn set_symbol(&mut self, symbol: i8, all_states: Option<&mut AllStates>);
    fn act(&mut self, state: &mut State, all_states: &AllStates) -> (usize, usize, i8);
}

// 修改 Player 实现
impl PlayerBehavior for Player {
    fn reset(&mut self) {
        self.reset();
    }

    fn set_state(&mut self, state: &mut State) {
        self.set_state(state);
    }

    fn set_symbol(&mut self, symbol: i8, all_states: Option<&mut AllStates>) {
        if let Some(states) = all_states {
            Player::set_symbol(self, symbol, states);
        } else {
            self.symbol = symbol;
        }
    }

    fn act(&mut self, state: &mut State, all_states: &AllStates) -> (usize, usize, i8) {
        self.act(state, all_states)
    }
}

// 修改 HumanPlayer 实现
impl PlayerBehavior for HumanPlayer {
    fn reset(&mut self) {
        self.reset();
    }

    fn set_state(&mut self, state: &mut State) {
        self.set_state(state);
    }

    fn set_symbol(&mut self, symbol: i8, _all_states: Option<&mut AllStates>) {
        HumanPlayer::set_symbol(self, symbol);
    }

    fn act(&mut self, _state: &mut State, _all_states: &AllStates) -> (usize, usize, i8) {
        // 不传递参数，因为HumanPlayer::act()不需要参数
        // 但确保self.state已经被设置
        HumanPlayer::act(self)
    }
}

enum GameOrder {
    HumanFirst(Judger<HumanPlayer, Player>),
    AIFirst(Judger<Player, HumanPlayer>),
}

impl GameOrder {
    fn play(&mut self, print_state: bool) -> i8 {
        match self {
            GameOrder::HumanFirst(judger) => judger.play(print_state),
            GameOrder::AIFirst(judger) => judger.play(print_state),
        }
    }
}

fn train(epochs_milestones: &[usize], print_every_n: usize) -> io::Result<()> {
    println!("开始训练，将在以下迭代次数保存模型: {:?}", epochs_milestones);
    
    let all_states = get_all_states();
    
    // 使用固定的探索率
    let epsilon = 0.01;  // 固定的探索率，不再衰减
    
    let player1 = Player::new(0.1, epsilon);
    let player2 = Player::new(0.1, epsilon);
    
    let mut judger = Judger::new(player1, player2, all_states);
    
    let mut player1_win = 0.0;
    let mut player2_win = 0.0;
    
    // 对每个训练里程碑进行排序
    let mut sorted_milestones = epochs_milestones.to_vec();
    sorted_milestones.sort_unstable();
    
    let mut next_milestone_idx = 0;
    let max_epochs = *sorted_milestones.last().unwrap_or(&100_000);
    
    for i in 1..=max_epochs {
        // 移除了epsilon衰减的代码
        
        let winner = judger.play(false);
        
        if winner == 1 {
            player1_win += 1.0;
        }
        if winner == -1 {
            player2_win += 1.0;
        }
        
        if i % print_every_n == 0 {
            println!(
                "Epoch {}/{}, epsilon: {:.5}, player 1 winrate: {:.2}, player 2 winrate: {:.2}",
                i, max_epochs, epsilon, player1_win / i as f64, player2_win / i as f64
            );
        }
        
        // 检查是否到达了一个里程碑
        if next_milestone_idx < sorted_milestones.len() && i == sorted_milestones[next_milestone_idx] {
            // 保存当前模型
            let milestone = sorted_milestones[next_milestone_idx];
            println!("已达到 {} 次迭代里程碑，保存模型...", milestone);
            
            judger.p1.save_policy_with_name(&format!("policy_first_{}.bin", milestone))?;
            judger.p2.save_policy_with_name(&format!("policy_second_{}.bin", milestone))?;
            
            next_milestone_idx += 1;
        }
        
        judger.p1.backup();
        judger.p2.backup();
        judger.reset();
    }
    
    // 最后一次保存使用默认名称
    judger.p1.save_policy()?;
    judger.p2.save_policy()?;
    
    Ok(())
}

fn model_tournament() -> io::Result<()> {
    println!("正在进行AI模型锦标赛...");
    
    let all_states = get_all_states();
    
    // 定义所有模型的迭代次数
    let milestones = [1_000, 10_000, 100_000, 1_000_000, 10_000_000];
    
    // 创建锦标赛结果矩阵
    let mut results = vec![vec![0; milestones.len()]; milestones.len()];
    
    // 每对模型比赛的次数
    let matches_per_pair = 100;
    
    // 为每一对模型进行对战
    for (i, &milestone_i) in milestones.iter().enumerate() {
        for (j, &milestone_j) in milestones.iter().enumerate() {            
            println!("模型 {} 次训练 vs 模型 {} 次训练...", milestone_i, milestone_j);
            
            // 加载两个模型
            let mut player1 = Player::new(0.1, 0.0);
            player1.symbol = 1;
            player1.load_policy_with_name(&format!("policy_first_{}.bin", milestone_i))?;
            
            let mut player2 = Player::new(0.1, 0.0);
            player2.symbol = -1;
            player2.load_policy_with_name(&format!("policy_second_{}.bin", milestone_j))?;
            
            let mut judger = Judger::new(player1, player2, all_states.clone());
            
            let mut model_i_wins = 0;
            let mut model_j_wins = 0;
            let mut ties = 0;
            
            for _ in 0..matches_per_pair {
                let winner = judger.play(false);
                
                if winner == 1 {
                    model_i_wins += 1;
                } else if winner == -1 {
                    model_j_wins += 1;
                } else {
                    ties += 1;
                }
                
                judger.reset();
            }
            
            results[i][j] = model_i_wins;
            
            println!("结果: {} 胜 {} 负 {} 平", model_i_wins, model_j_wins, ties);
        }
    }
    
    // 打印锦标赛结果表格
    println!("\n锦标赛结果 (行击败列的胜场数，每对模型对战{}次):", matches_per_pair);
    
    // 打印表头
    print!("{:>12}", "");
    for &milestone in &milestones {
        print!("{:>12}", milestone);
    }
    println!();
    
    // 打印结果
    for (i, &milestone_i) in milestones.iter().enumerate() {
        print!("{:>12}", milestone_i);
        for j in 0..milestones.len() {
            if i == j {
                print!("{:>12}", "-");
            } else {
                print!("{:>12}", results[i][j]);
            }
        }
        println!();
    }
    
    Ok(())
}

fn play() -> io::Result<()> {
    println!("正在加载AI模型和游戏状态，请稍候...");

    // 只生成一次所有状态
    let all_states = get_all_states();
    
    // 定义可用的模型迭代次数
    let milestones = [1_000, 10_000, 100_000, 1_000_000, 10_000_000];
    
    // 让玩家选择AI模型
    println!("请选择AI模型训练迭代次数：");
    for (i, &milestone) in milestones.iter().enumerate() {
        println!("[{}] {} 次迭代", i+1, milestone);
    }
    
    let mut input = String::new();
    io::stdin().read_line(&mut input).expect("Failed to read input");
    
    let milestone_idx = match input.trim().parse::<usize>() {
        Ok(idx) if idx >= 1 && idx <= milestones.len() => idx - 1,
        _ => {
            println!("无效选择，使用最后一个模型（{}次迭代）", milestones.last().unwrap());
            milestones.len() - 1
        }
    };
    let milestone = milestones[milestone_idx];
    
    println!("使用{}次迭代训练的AI模型", milestone);
    
    // 预先加载两种AI策略（先手和后手）
    let mut ai_first = Player::new(0.1, 0.0);
    ai_first.symbol = 1; // 先手
    ai_first.load_policy_with_name(&format!("policy_first_{}.bin", milestone))?;
    
    let mut ai_second = Player::new(0.1, 0.0);
    ai_second.symbol = -1; // 后手
    ai_second.load_policy_with_name(&format!("policy_second_{}.bin", milestone))?;

    println!("加载完成！游戏开始！");
    println!("井字棋游戏: 需要连成 {} 子才能获胜", WIN_COUNT);
    
    // 以下是原有的游戏循环，保持不变
    // 记录游戏统计
    let mut wins = 0;
    let mut losses = 0;
    let mut ties = 0;
    
    loop {
        // 让玩家选择难度
        println!("请选择游戏难度: [1] 简单  [2] 中等  [3] 困难");
        let mut input = String::new();
        io::stdin().read_line(&mut input).expect("Failed to read input");

        let epsilon = match input.trim() {
            "1" => 0.3,  // 简单：AI有30%概率随机走子
            "2" => 0.1,  // 中等：AI有10%概率随机走子
            "3" => 0.0,  // 困难：AI始终选择最优策略
            _ => 0.1     // 默认中等难度
        };

        // 让玩家选择先手或后手
        println!("请选择: [1] 玩家先手(*)  [2] AI先手(x)");
        let mut input = String::new();
        io::stdin().read_line(&mut input).expect("Failed to read input");
        
        let player_first = match input.trim() {
            "1" => true,
            "2" => false,
            _ => {
                println!("无效选择，默认玩家先手");
                true
            }
        };
        
        // 创建人类玩家
        let human_player = HumanPlayer::new();
        
        // 根据玩家选择决定先后手，并克隆适当的AI
        let mut game_order = if player_first {
            println!("您选择了先手，您的棋子是 * (先手)");
            let mut ai = ai_second.clone();
            ai.epsilon = epsilon;
            GameOrder::HumanFirst(Judger::new(human_player, ai, all_states.clone()))
        } else {
            println!("您选择了后手，您的棋子是 x (后手)");
            let mut ai = ai_first.clone();
            ai.epsilon = epsilon;
            GameOrder::AIFirst(Judger::new(ai, human_player, all_states.clone()))
        };

        let winner = game_order.play(true);
        
        // 根据先后手情况正确显示游戏结果
        if player_first {
            if winner == 1 {
                println!("恭喜，您赢了！");
                wins += 1;
            } else if winner == -1 {
                println!("很遗憾，您输了！");
                losses += 1;
            } else {
                println!("平局！");
                ties += 1;
            }
        } else {
            if winner == -1 {
                println!("恭喜，您赢了！");
                wins += 1;
            } else if winner == 1 {
                println!("很遗憾，您输了！");
                losses += 1;
            } else {
                println!("平局！");
                ties += 1;
            }
        }
        
        // 显示当前战绩
        println!("您的当前战绩: {}胜 {}负 {}平", wins, losses, ties);
        
        println!("再来一局？ (y/n)");
        let mut input = String::new();
        io::stdin().read_line(&mut input).expect("Failed to read input");
        
        if input.trim().to_lowercase() != "y" {
            println!("感谢您的参与！最终战绩: {}胜 {}负 {}平", wins, losses, ties);
            break;
        }
    }
    
    Ok(())
}

fn main() -> io::Result<()> {
    println!("井字棋强化学习系统");
    
    // 定义训练里程碑
    let milestones = [1_000, 10_000, 100_000, 1_000_000, 10_000_000];
    
    // 检查是否已经训练了所有模型
    let all_models_exist = milestones.iter().all(|&m| {
        Path::new(&format!("policy_first_{}.bin", m)).exists() && 
        Path::new(&format!("policy_second_{}.bin", m)).exists()
    });
    
    if !all_models_exist {
        println!("需要训练和保存所有模型...");
        train(&milestones, 10000)?;
    } else {
        println!("所有模型已存在，无需重新训练。");
    }
    
    loop {
        println!("\n请选择：");
        println!("[1] 与AI进行游戏");
        println!("[2] 进行AI模型锦标赛");
        println!("[3] 重新训练所有模型");
        println!("[0] 退出");
        
        let mut input = String::new();
        io::stdin().read_line(&mut input).expect("Failed to read input");
        
        match input.trim() {
            "1" => {
                // 开始人机对战
                play()?;
            },
            "2" => {
                // 进行模型锦标赛
                model_tournament()?;
            },
            "3" => {
                // 重新训练所有模型
                train(&milestones, 10000)?;
            },
            "0" => {
                println!("谢谢使用，再见！");
                break;
            },
            _ => {
                println!("无效选择，请重试。");
            }
        }
    }
    
    Ok(())
}