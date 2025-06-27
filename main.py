import numpy as np

nInst = 50
currentPos = np.zeros(nInst)

class ImprovedMomentumStrategy:
    def __init__(self):
        # Momentum parameters
        self.momentum_window = 20       # 20-day momentum
        self.short_ma_window = 10       # Short moving average
        self.long_ma_window = 30        # Long moving average
        
        # RSI parameters
        self.rsi_window = 14
        self.rsi_overbought = 70
        self.rsi_oversold = 30
        
        # Trading parameters
        self.position_size = 3000       # Position size per trade
        self.max_positions = 5          # Allow more positions
        self.min_volume_filter = 100    # Minimum average volume
        
        # Risk management
        self.stop_loss_pct = 0.04       # 4% stop loss
        self.take_profit_pct = 0.08     # 8% take profit
        self.trailing_stop_pct = 0.03   # 3% trailing stop
        
        # Entry thresholds
        self.min_momentum_long = 0.02   # 2% momentum for long
        self.max_momentum_short = -0.02 # -2% momentum for short
        self.min_volatility = 0.005     # Minimum volatility to trade
        self.max_volatility = 0.06      # Maximum volatility to trade
        
        # Position tracking
        self.entry_prices = {}
        self.entry_types = {}  # 'long' or 'short'
        self.highest_prices = {}
        self.lowest_prices = {}
        self.entry_days = {}
        
        # Performance tracking
        self.trades = []
        
    def calculate_rsi(self, prices):
        """Calculate RSI for a price series"""
        if len(prices) < self.rsi_window + 1:
            return 50  # Neutral RSI if not enough data
        
        deltas = np.diff(prices)
        seed = deltas[:self.rsi_window+1]
        up = seed[seed >= 0].sum() / self.rsi_window
        down = -seed[seed < 0].sum() / self.rsi_window
        
        if down == 0:
            return 100
        
        rs = up / down
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_volatility(self, prices):
        """Calculate annualized volatility"""
        if len(prices) < 2:
            return 0.02
        
        returns = np.diff(prices) / prices[:-1]
        return np.std(returns) * np.sqrt(252)
    
    def calculate_signals(self, prices):
        """Calculate trading signals for all stocks"""
        n_inst, n_days = prices.shape
        
        if n_days < self.long_ma_window + 1:
            return {}
        
        signals = {}
        
        for i in range(n_inst):
            try:
                # Get price data
                price_series = prices[i, :]
                current_price = price_series[-1]
                
                # Skip if price is too low (likely bad data)
                if current_price < 1:
                    continue
                
                # Calculate indicators
                short_ma = np.mean(price_series[-self.short_ma_window:])
                long_ma = np.mean(price_series[-self.long_ma_window:])
                
                # Momentum
                momentum_price = price_series[-self.momentum_window]
                momentum = (current_price - momentum_price) / momentum_price
                
                # RSI
                rsi = self.calculate_rsi(price_series[-self.rsi_window-1:])
                
                # Volatility
                volatility = self.calculate_volatility(price_series[-20:])
                
                # Volume proxy (price changes as volume indicator)
                volume_proxy = np.std(np.diff(price_series[-10:]))
                
                # Trend strength
                trend_strength = (short_ma - long_ma) / long_ma
                
                signals[i] = {
                    'momentum': momentum,
                    'rsi': rsi,
                    'volatility': volatility,
                    'trend_strength': trend_strength,
                    'short_ma': short_ma,
                    'long_ma': long_ma,
                    'volume_proxy': volume_proxy,
                    'current_price': current_price
                }
                
            except Exception as e:
                continue
        
        return signals
    
    def generate_entry_signals(self, signals, current_positions):
        """Generate entry signals for both long and short positions"""
        long_signals = []
        short_signals = []
        
        for stock_idx, signal in signals.items():
            # Skip if already have position
            if stock_idx in current_positions:
                continue
            
            # Skip if volatility out of range
            if signal['volatility'] < self.min_volatility or signal['volatility'] > self.max_volatility:
                continue
            
            # Long signal conditions
            if (signal['momentum'] > self.min_momentum_long and
                signal['rsi'] < 65 and  # Not overbought
                signal['trend_strength'] > 0.01 and  # Upward trend
                signal['short_ma'] > signal['long_ma']):  # MA crossover
                
                score = (signal['momentum'] * 2 + 
                        (100 - signal['rsi']) / 100 + 
                        signal['trend_strength'] * 5 +
                        (1 / signal['volatility']) * 0.1)
                
                long_signals.append((stock_idx, score, signal))
            
            # Short signal conditions
            elif (signal['momentum'] < self.max_momentum_short and
                  signal['rsi'] > 35 and  # Not oversold yet
                  signal['trend_strength'] < -0.01 and  # Downward trend
                  signal['short_ma'] < signal['long_ma']):  # MA crossover
                
                score = (abs(signal['momentum']) * 2 + 
                        signal['rsi'] / 100 + 
                        abs(signal['trend_strength']) * 5 +
                        (1 / signal['volatility']) * 0.1)
                
                short_signals.append((stock_idx, score, signal))
        
        # Sort by score
        long_signals.sort(key=lambda x: x[1], reverse=True)
        short_signals.sort(key=lambda x: x[1], reverse=True)
        
        return long_signals, short_signals
    
    def check_exits(self, prices, signals, current_day):
        """Check exit conditions for all positions"""
        exits = []
        current_prices = prices[:, -1]
        
        for stock_idx in list(self.entry_prices.keys()):
            if stock_idx not in self.entry_prices:
                continue
                
            current_price = current_prices[stock_idx]
            entry_price = self.entry_prices[stock_idx]
            position_type = self.entry_types[stock_idx]
            
            # Calculate return based on position type
            if position_type == 'long':
                return_pct = (current_price - entry_price) / entry_price
                
                # Update highest price
                if stock_idx not in self.highest_prices:
                    self.highest_prices[stock_idx] = entry_price
                self.highest_prices[stock_idx] = max(self.highest_prices[stock_idx], current_price)
                
                # Exit conditions for long positions
                if return_pct >= self.take_profit_pct:
                    exits.append((stock_idx, f"Long TP: +{return_pct*100:.1f}%"))
                elif return_pct <= -self.stop_loss_pct:
                    exits.append((stock_idx, f"Long SL: {return_pct*100:.1f}%"))
                elif current_price < self.highest_prices[stock_idx] * (1 - self.trailing_stop_pct):
                    exits.append((stock_idx, f"Long TS: {return_pct*100:.1f}%"))
                elif stock_idx in signals and signals[stock_idx]['rsi'] > self.rsi_overbought:
                    exits.append((stock_idx, f"Long RSI exit: {return_pct*100:.1f}%"))
                
            else:  # short position
                return_pct = (entry_price - current_price) / entry_price
                
                # Update lowest price
                if stock_idx not in self.lowest_prices:
                    self.lowest_prices[stock_idx] = entry_price
                self.lowest_prices[stock_idx] = min(self.lowest_prices[stock_idx], current_price)
                
                # Exit conditions for short positions
                if return_pct >= self.take_profit_pct:
                    exits.append((stock_idx, f"Short TP: +{return_pct*100:.1f}%"))
                elif return_pct <= -self.stop_loss_pct:
                    exits.append((stock_idx, f"Short SL: {return_pct*100:.1f}%"))
                elif current_price > self.lowest_prices[stock_idx] * (1 + self.trailing_stop_pct):
                    exits.append((stock_idx, f"Short TS: {return_pct*100:.1f}%"))
                elif stock_idx in signals and signals[stock_idx]['rsi'] < self.rsi_oversold:
                    exits.append((stock_idx, f"Short RSI exit: {return_pct*100:.1f}%"))
            
            # Time-based exit (30 days max)
            if current_day - self.entry_days[stock_idx] > 30:
                exits.append((stock_idx, f"Time exit: {return_pct*100:.1f}%"))
        
        return exits

# Global strategy instance
strategy = ImprovedMomentumStrategy()

def getMyPosition(prcSoFar):
    global currentPos, strategy
    
    (nins, nt) = prcSoFar.shape
    
    # Need enough data for indicators
    if nt < strategy.long_ma_window + 2:
        return np.zeros(nins)
    
    current_prices = prcSoFar[:, -1]
    
    # Calculate signals for all stocks
    signals = strategy.calculate_signals(prcSoFar)
    
    # Check exits first
    exits = strategy.check_exits(prcSoFar, signals, nt)
    for stock_idx, reason in exits:
        if currentPos[stock_idx] != 0:
            # Record trade
            position_type = strategy.entry_types.get(stock_idx, 'long')
            entry_price = strategy.entry_prices.get(stock_idx, current_prices[stock_idx])
            exit_price = current_prices[stock_idx]
            
            if position_type == 'long':
                pnl = (exit_price - entry_price) / entry_price
            else:
                pnl = (entry_price - exit_price) / entry_price
            
            strategy.trades.append({
                'day': nt,
                'stock': stock_idx,
                'type': position_type,
                'pnl': pnl,
                'reason': reason
            })
            
            print(f"Day {nt}: EXIT {reason}")
            
            # Clear position
            currentPos[stock_idx] = 0
            if stock_idx in strategy.entry_prices:
                del strategy.entry_prices[stock_idx]
            if stock_idx in strategy.entry_types:
                del strategy.entry_types[stock_idx]
            if stock_idx in strategy.highest_prices:
                del strategy.highest_prices[stock_idx]
            if stock_idx in strategy.lowest_prices:
                del strategy.lowest_prices[stock_idx]
            if stock_idx in strategy.entry_days:
                del strategy.entry_days[stock_idx]
    
    # Get current positions
    current_positions = {i for i in range(nins) if currentPos[i] != 0}
    num_positions = len(current_positions)
    
    # Generate entry signals
    long_signals, short_signals = strategy.generate_entry_signals(signals, current_positions)
    
    # Enter new positions if we have capacity
    if num_positions < strategy.max_positions:
        # Try to balance long and short positions
        all_signals = []
        
        # Add best long signals
        for sig in long_signals[:3]:
            all_signals.append(('long', sig))
        
        # Add best short signals
        for sig in short_signals[:3]:
            all_signals.append(('short', sig))
        
        # Sort all signals by score
        all_signals.sort(key=lambda x: x[1][1], reverse=True)
        
        # Enter positions
        for position_type, (stock_idx, score, signal) in all_signals:
            if num_positions >= strategy.max_positions:
                break
            
            # Calculate position size
            if position_type == 'long':
                shares = int(strategy.position_size / current_prices[stock_idx])
                if shares > 0:
                    currentPos[stock_idx] = shares
                    strategy.entry_prices[stock_idx] = current_prices[stock_idx]
                    strategy.entry_types[stock_idx] = 'long'
                    strategy.entry_days[stock_idx] = nt
                    num_positions += 1
                    
                    print(f"Day {nt}: LONG Stock {stock_idx}: {shares} shares at ${current_prices[stock_idx]:.2f}, "
                          f"Mom: {signal['momentum']:.3f}, RSI: {signal['rsi']:.1f}, Score: {score:.3f}")
            
            else:  # short position
                shares = int(strategy.position_size / current_prices[stock_idx])
                if shares > 0:
                    currentPos[stock_idx] = -shares
                    strategy.entry_prices[stock_idx] = current_prices[stock_idx]
                    strategy.entry_types[stock_idx] = 'short'
                    strategy.entry_days[stock_idx] = nt
                    num_positions += 1
                    
                    print(f"Day {nt}: SHORT Stock {stock_idx}: {shares} shares at ${current_prices[stock_idx]:.2f}, "
                          f"Mom: {signal['momentum']:.3f}, RSI: {signal['rsi']:.1f}, Score: {score:.3f}")
    
    # Performance reporting
    if nt % 50 == 0 and len(strategy.trades) > 0:
        wins = sum(1 for t in strategy.trades if t['pnl'] > 0)
        total_pnl = sum(t['pnl'] for t in strategy.trades)
        avg_pnl = total_pnl / len(strategy.trades)
        win_rate = wins / len(strategy.trades) * 100
        
        print(f"\nDay {nt} Performance:")
        print(f"Trades: {len(strategy.trades)}, Win Rate: {win_rate:.1f}%, Avg PnL: {avg_pnl*100:.2f}%")
        print(f"Current Positions: {num_positions}")
        print("-" * 50)
    
    return currentPos.astype(int)