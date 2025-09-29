#!/usr/bin/env python3
"""
Comprehensive stress testing suite for Flappy Bird DQN Transformer
This covers all the traditional testing approaches used before AI code review tools.
"""

import torch
import torch.nn as nn
import numpy as np
import time
import psutil
import os
import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Import your modules
from model import DQN, BuildEncoder
from experience_replay import Replay
from agent import Agent
import flappy_bird_gymnasium
import gymnasium

class StressTester:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.test_results = {}
        self.performance_metrics = {}
        
    def run_all_tests(self):
        """Run all stress tests and generate report"""
        print("🧪 Starting Comprehensive Stress Testing Suite")
        print("=" * 60)
        
        tests = [
            ("Unit Tests", self.test_unit_components),
            ("Gradient Flow", self.test_gradient_flow),
            ("Numerical Stability", self.test_numerical_stability),
            ("Edge Cases", self.test_edge_cases),
            ("Memory Management", self.test_memory_management),
            ("Performance Benchmarks", self.test_performance),
            ("Convergence Validation", self.test_convergence),
            ("Hyperparameter Sensitivity", self.test_hyperparameter_sensitivity)
        ]
        
        for test_name, test_func in tests:
            print(f"\n🔍 Running {test_name}...")
            try:
                result = test_func()
                self.test_results[test_name] = result
                status = "✅ PASSED" if result['passed'] else "❌ FAILED"
                print(f"{status} - {test_name}")
                if not result['passed']:
                    print(f"   Issues: {result['issues']}")
            except Exception as e:
                print(f"❌ FAILED - {test_name}: {str(e)}")
                self.test_results[test_name] = {'passed': False, 'issues': [str(e)]}
        
        self.generate_report()
    
    def test_unit_components(self):
        """Test individual components in isolation"""
        issues = []
        
        # Test 1: Model instantiation and forward pass
        try:
            model = DQN(8, 2, 256, 256, 128, 512, 8, 6, 0.1, 16).to(self.device)
            dummy_input = torch.randn(1, 16, 8).to(self.device)
            output = model(dummy_input)
            assert output.shape == (1, 2), f"Expected output shape (1, 2), got {output.shape}"
        except Exception as e:
            issues.append(f"Model instantiation/forward pass failed: {e}")
        
        # Test 2: Replay buffer basic operations
        try:
            replay = Replay(1000)
            # Test empty buffer
            assert replay.size == 0, "Empty buffer should have size 0"
            
            # Test adding transitions
            dummy_transitions = [(torch.randn(8), torch.tensor(0), torch.randn(8), torch.tensor(1.0), False) for _ in range(5)]
            replay.push(dummy_transitions)
            assert replay.size == 5, f"Expected size 5, got {replay.size}"
        except Exception as e:
            issues.append(f"Replay buffer operations failed: {e}")
        
        # Test 3: Agent initialization
        try:
            agent = Agent("flappybird1")
            assert hasattr(agent, 'device'), "Agent should have device attribute"
            assert hasattr(agent, 'episode_rewards'), "Agent should have episode_rewards"
        except Exception as e:
            issues.append(f"Agent initialization failed: {e}")
        
        return {'passed': len(issues) == 0, 'issues': issues}
    
    def test_gradient_flow(self):
        """Test gradient flow and backpropagation"""
        issues = []
        
        try:
            model = DQN(8, 2, 256, 256, 128, 512, 8, 6, 0.1, 16).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            # Test gradient flow
            dummy_input = torch.randn(32, 16, 8).to(self.device)
            dummy_target = torch.randn(32, 2).to(self.device)
            
            output = model(dummy_input)
            loss = nn.MSELoss()(output, dummy_target)
            loss.backward()
            
            # Check for gradient issues
            total_norm = 0
            param_count = 0
            for param in model.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    param_count += 1
                    
                    # Check for NaN or Inf gradients
                    if torch.isnan(param.grad).any():
                        issues.append("NaN gradients detected")
                    if torch.isinf(param.grad).any():
                        issues.append("Infinite gradients detected")
            
            total_norm = total_norm ** (1. / 2)
            
            if total_norm == 0:
                issues.append("All gradients are zero - possible vanishing gradient problem")
            elif total_norm > 100:
                issues.append(f"Very large gradient norm: {total_norm:.2f} - possible exploding gradients")
            
            # Test gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
        except Exception as e:
            issues.append(f"Gradient flow test failed: {e}")
        
        return {'passed': len(issues) == 0, 'issues': issues}
    
    def test_numerical_stability(self):
        """Test numerical stability with extreme values"""
        issues = []
        
        try:
            model = DQN(8, 2, 256, 256, 128, 512, 8, 6, 0.1, 16).to(self.device)
            
            # Test with very small values
            small_input = torch.randn(1, 16, 8).to(self.device) * 1e-6
            output_small = model(small_input)
            if torch.isnan(output_small).any() or torch.isinf(output_small).any():
                issues.append("Model unstable with very small inputs")
            
            # Test with very large values
            large_input = torch.randn(1, 16, 8).to(self.device) * 1e6
            output_large = model(large_input)
            if torch.isnan(output_large).any() or torch.isinf(output_large).any():
                issues.append("Model unstable with very large inputs")
            
            # Test with zero input
            zero_input = torch.zeros(1, 16, 8).to(self.device)
            output_zero = model(zero_input)
            if torch.isnan(output_zero).any() or torch.isinf(output_zero).any():
                issues.append("Model unstable with zero inputs")
            
            # Test attention mechanism stability
            encoder = BuildEncoder(8, 16, 0.1, 128, 512, 8, 6).to(self.device)
            test_input = torch.randn(1, 16, 8).to(self.device)
            encoder_output = encoder(test_input)
            
            # Check for reasonable output ranges
            if torch.abs(encoder_output).max() > 100:
                issues.append(f"Encoder output values too large: {torch.abs(encoder_output).max()}")
            
        except Exception as e:
            issues.append(f"Numerical stability test failed: {e}")
        
        return {'passed': len(issues) == 0, 'issues': issues}
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        issues = []
        
        try:
            # Test 1: Empty replay buffer sampling
            replay = Replay(1000)
            try:
                batch = replay.sample(32, 8, 16, self.device)
                issues.append("Should not be able to sample from empty buffer")
            except:
                pass  # Expected behavior
            
            # Test 2: Single transition in replay buffer
            replay = Replay(1000)
            single_transition = [(torch.randn(8), torch.tensor(0), torch.randn(8), torch.tensor(1.0), False)]
            replay.push(single_transition)
            batch = replay.sample(1, 8, 16, self.device)
            assert batch[0].shape == (1, 16, 8), f"Expected shape (1, 16, 8), got {batch[0].shape}"
            
            # Test 3: Sequence length edge cases
            model = DQN(8, 2, 256, 256, 128, 512, 8, 6, 0.1, 16).to(self.device)
            
            # Test with sequence length 1
            short_input = torch.randn(1, 1, 8).to(self.device)
            try:
                output = model(short_input)
            except Exception as e:
                issues.append(f"Model failed with sequence length 1: {e}")
            
            # Test 4: Batch size edge cases
            single_batch = torch.randn(1, 16, 8).to(self.device)
            large_batch = torch.randn(1000, 16, 8).to(self.device)
            
            try:
                model(single_batch)
                model(large_batch)
            except Exception as e:
                issues.append(f"Model failed with edge case batch sizes: {e}")
            
            # Test 5: Action space edge cases
            env = gymnasium.make("FlappyBird-v0", render_mode=None, use_lidar=True)
            state_dim = env.observation_space.shape[0]
            num_actions = env.action_space.n
            
            # Test with invalid action
            try:
                env.step(999)  # Invalid action
                issues.append("Environment should reject invalid actions")
            except:
                pass  # Expected behavior
            
        except Exception as e:
            issues.append(f"Edge case test failed: {e}")
        
        return {'passed': len(issues) == 0, 'issues': issues}
    
    def test_memory_management(self):
        """Test memory usage and potential leaks"""
        issues = []
        
        try:
            # Test memory usage during training
            initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            model = DQN(8, 2, 256, 256, 128, 512, 8, 6, 0.1, 16).to(self.device)
            replay = Replay(10000)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            # Simulate training loop
            for i in range(100):
                # Generate dummy data
                transitions = []
                for _ in range(10):
                    transitions.append((
                        torch.randn(8).to(self.device),
                        torch.tensor(np.random.randint(0, 2)).to(self.device),
                        torch.randn(8).to(self.device),
                        torch.tensor(np.random.randn()).to(self.device),
                        np.random.choice([True, False])
                    ))
                
                replay.push(transitions)
                
                if replay.size > 64:
                    batch = replay.sample(32, 8, 16, self.device)
                    
                    # Forward pass
                    states = batch[0]
                    actions = batch[1]
                    rewards = batch[2]
                    next_states = batch[3]
                    dones = batch[4]
                    
                    # Compute loss
                    with torch.no_grad():
                        next_actions = model(next_states).argmax(dim=1)
                        target_q = model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                        new_q = rewards + (1-dones) * (target_q * 0.99)
                    
                    policy_q = model(states).gather(1, actions.unsqueeze(1))
                    loss = nn.MSELoss()(policy_q, new_q.unsqueeze(1))
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                # Check memory every 20 iterations
                if i % 20 == 0:
                    current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                    memory_increase = current_memory - initial_memory
                    
                    if memory_increase > 500:  # More than 500MB increase
                        issues.append(f"Potential memory leak: {memory_increase:.1f}MB increase after {i} iterations")
            
            # Test GPU memory if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                initial_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
                
                # Run some operations
                for _ in range(50):
                    dummy = torch.randn(100, 16, 8).to(self.device)
                    output = model(dummy)
                    del dummy
                
                torch.cuda.empty_cache()
                final_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
                
                if final_gpu_memory > initial_gpu_memory + 100:  # More than 100MB increase
                    issues.append(f"Potential GPU memory leak: {final_gpu_memory - initial_gpu_memory:.1f}MB increase")
            
        except Exception as e:
            issues.append(f"Memory management test failed: {e}")
        
        return {'passed': len(issues) == 0, 'issues': issues}
    
    def test_performance(self):
        """Test performance benchmarks"""
        issues = []
        performance_data = {}
        
        try:
            model = DQN(8, 2, 256, 256, 128, 512, 8, 6, 0.1, 16).to(self.device)
            replay = Replay(10000)
            
            # Benchmark forward pass
            dummy_input = torch.randn(64, 16, 8).to(self.device)
            
            # Warm up
            for _ in range(10):
                _ = model(dummy_input)
            
            # Time forward pass
            start_time = time.time()
            for _ in range(100):
                _ = model(dummy_input)
            forward_time = (time.time() - start_time) / 100 * 1000  # ms per forward pass
            
            performance_data['forward_pass_ms'] = forward_time
            
            if forward_time > 50:  # More than 50ms per forward pass
                issues.append(f"Slow forward pass: {forward_time:.2f}ms per batch")
            
            # Benchmark replay buffer sampling
            # Fill buffer
            for _ in range(100):
                transitions = []
                for _ in range(10):
                    transitions.append((
                        torch.randn(8).to(self.device),
                        torch.tensor(np.random.randint(0, 2)).to(self.device),
                        torch.randn(8).to(self.device),
                        torch.tensor(np.random.randn()).to(self.device),
                        np.random.choice([True, False])
                    ))
                replay.push(transitions)
            
            # Time sampling
            start_time = time.time()
            for _ in range(100):
                _ = replay.sample(32, 8, 16, self.device)
            sampling_time = (time.time() - start_time) / 100 * 1000  # ms per sample
            
            performance_data['sampling_ms'] = sampling_time
            
            if sampling_time > 20:  # More than 20ms per sample
                issues.append(f"Slow replay buffer sampling: {sampling_time:.2f}ms per batch")
            
            # Benchmark full training step
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            start_time = time.time()
            for _ in range(50):
                batch = replay.sample(32, 8, 16, self.device)
                
                states = batch[0]
                actions = batch[1]
                rewards = batch[2]
                next_states = batch[3]
                dones = batch[4]
                
                with torch.no_grad():
                    next_actions = model(next_states).argmax(dim=1)
                    target_q = model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                    new_q = rewards + (1-dones) * (target_q * 0.99)
                
                policy_q = model(states).gather(1, actions.unsqueeze(1))
                loss = nn.MSELoss()(policy_q, new_q.unsqueeze(1))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            training_time = (time.time() - start_time) / 50 * 1000  # ms per training step
            performance_data['training_step_ms'] = training_time
            
            if training_time > 100:  # More than 100ms per training step
                issues.append(f"Slow training step: {training_time:.2f}ms per step")
            
            self.performance_metrics = performance_data
            
        except Exception as e:
            issues.append(f"Performance test failed: {e}")
        
        return {'passed': len(issues) == 0, 'issues': issues, 'performance': performance_data}
    
    def test_convergence(self):
        """Test learning convergence with short training runs"""
        issues = []
        
        try:
            # Create a simple test environment
            model = DQN(8, 2, 256, 256, 128, 512, 8, 6, 0.1, 16).to(self.device)
            target_model = DQN(8, 2, 256, 256, 128, 512, 8, 6, 0.1, 16).to(self.device)
            target_model.load_state_dict(model.state_dict())
            
            replay = Replay(5000)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            losses = []
            q_values = []
            
            # Generate some training data
            for episode in range(50):
                # Generate random episode
                episode_transitions = []
                for step in range(20):
                    state = torch.randn(8).to(self.device)
                    action = torch.tensor(np.random.randint(0, 2)).to(self.device)
                    next_state = torch.randn(8).to(self.device)
                    reward = torch.tensor(np.random.randn()).to(self.device)
                    done = step == 19  # End episode at step 19
                    
                    episode_transitions.append((state, action, next_state, reward, done))
                
                replay.push(episode_transitions)
                
                # Train if we have enough data
                if replay.size > 64:
                    batch = replay.sample(32, 8, 16, self.device)
                    
                    states = batch[0]
                    actions = batch[1]
                    rewards = batch[2]
                    next_states = batch[3]
                    dones = batch[4]
                    
                    with torch.no_grad():
                        next_actions = model(next_states).argmax(dim=1)
                        target_q = target_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                        new_q = rewards + (1-dones) * (target_q * 0.99)
                    
                    policy_q = model(states).gather(1, actions.unsqueeze(1))
                    loss = nn.MSELoss()(policy_q, new_q.unsqueeze(1))
                    
                    losses.append(loss.item())
                    q_values.append(policy_q.mean().item())
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    # Update target network
                    if episode % 10 == 0:
                        target_model.load_state_dict(model.state_dict())
            
            # Check convergence indicators
            if len(losses) > 10:
                recent_losses = losses[-10:]
                early_losses = losses[:10]
                
                # Loss should generally decrease
                if np.mean(recent_losses) > np.mean(early_losses) * 1.5:
                    issues.append("Loss not decreasing - possible learning issues")
                
                # Q-values should be reasonable
                if np.abs(np.mean(q_values)) > 100:
                    issues.append(f"Q-values too large: {np.mean(q_values):.2f}")
                
                # Check for NaN in losses
                if any(np.isnan(loss) for loss in losses):
                    issues.append("NaN values in loss - numerical instability")
            
        except Exception as e:
            issues.append(f"Convergence test failed: {e}")
        
        return {'passed': len(issues) == 0, 'issues': issues}
    
    def test_hyperparameter_sensitivity(self):
        """Test sensitivity to hyperparameter changes"""
        issues = []
        
        try:
            base_config = {
                'state_dim': 8, 'num_actions': 2, 'num_hidden_1': 256, 'num_hidden_2': 256,
                'd_model': 128, 'd_ff': 512, 'heads': 8, 'N': 6, 'dropout': 0.1, 'seq_len': 16
            }
            
            # Test different learning rates
            lr_results = {}
            for lr in [0.0001, 0.001, 0.01]:
                model = DQN(**base_config).to(self.device)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                
                dummy_input = torch.randn(32, 16, 8).to(self.device)
                dummy_target = torch.randn(32, 2).to(self.device)
                
                losses = []
                for _ in range(10):
                    output = model(dummy_input)
                    loss = nn.MSELoss()(output, dummy_target)
                    losses.append(loss.item())
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                lr_results[lr] = losses[-1]
            
            # Check if learning rate has reasonable effect
            if lr_results[0.0001] > lr_results[0.01] * 0.9:
                issues.append("Model not sensitive to learning rate changes")
            
            # Test different dropout rates
            dropout_results = {}
            for dropout in [0.0, 0.1, 0.5]:
                model = DQN(**{**base_config, 'dropout': dropout}).to(self.device)
                dummy_input = torch.randn(32, 16, 8).to(self.device)
                
                # Test multiple forward passes to see dropout effect
                outputs = []
                for _ in range(10):
                    with torch.no_grad():
                        output = model(dummy_input)
                        outputs.append(output)
                
                # Check variance in outputs (should be higher with dropout)
                output_variance = torch.var(torch.stack(outputs), dim=0).mean().item()
                dropout_results[dropout] = output_variance
            
            # Dropout should increase variance
            if dropout_results[0.5] < dropout_results[0.0] * 1.1:
                issues.append("Dropout not having expected effect on output variance")
            
        except Exception as e:
            issues.append(f"Hyperparameter sensitivity test failed: {e}")
        
        return {'passed': len(issues) == 0, 'issues': issues}
    
    def generate_report(self):
        """Generate comprehensive test report"""
        print("\n" + "=" * 60)
        print("📊 STRESS TEST REPORT")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result['passed'])
        
        print(f"Overall Status: {passed_tests}/{total_tests} tests passed")
        print(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
        
        print("\n📋 Detailed Results:")
        for test_name, result in self.test_results.items():
            status = "✅ PASSED" if result['passed'] else "❌ FAILED"
            print(f"  {status} {test_name}")
            if not result['passed'] and 'issues' in result:
                for issue in result['issues']:
                    print(f"    ⚠️  {issue}")
        
        if self.performance_metrics:
            print("\n⚡ Performance Metrics:")
            for metric, value in self.performance_metrics.items():
                print(f"  {metric}: {value:.2f}ms")
        
        # Generate recommendations
        print("\n💡 Recommendations:")
        recommendations = []
        
        for test_name, result in self.test_results.items():
            if not result['passed']:
                if "gradient" in test_name.lower():
                    recommendations.append("Consider gradient clipping and learning rate adjustment")
                elif "memory" in test_name.lower():
                    recommendations.append("Review memory management and consider reducing batch size")
                elif "performance" in test_name.lower():
                    recommendations.append("Optimize model architecture or reduce sequence length")
                elif "convergence" in test_name.lower():
                    recommendations.append("Check reward scaling and target network update frequency")
        
        if not recommendations:
            recommendations.append("All tests passed! Your model appears to be working correctly.")
        
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
        
        print("\n" + "=" * 60)

if __name__ == "__main__":
    tester = StressTester()
    tester.run_all_tests()
