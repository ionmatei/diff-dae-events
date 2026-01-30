
import json
import numpy as np
import yaml
import sys
import os
import itertools

# Add src to path
sys.path.append(os.getcwd())

from src.discrete_adjoint.dae_solver import DAESolver, AugmentedSolution

def verify_dae_output():
    report_path = 'debug/dae_structure_report.txt'
    
    # Load base config
    config_path = 'config/config_bouncing_ball.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    solver_cfg = config['dae_solver']
    dae_spec_file = solver_cfg['dae_specification_file']
    
    # Load DAE spec
    with open(dae_spec_file, 'r') as f:
        dae_data_template = json.load(f)
        
    # Parameters to test
    g_values = [9.8, 5.0, 15.0]
    e_values = [0.8, 0.5, 0.9]
    
    with open(report_path, 'w') as report:
        report.write("DAE Solver Structure Verification Report\n")
        report.write("========================================\n\n")
        
        all_passed = True
        
        for g_val, e_val in itertools.product(g_values, e_values):
            case_name = f"g={g_val}, e={e_val}"
            report.write(f"Testing Case: {case_name}\n")
            report.write("-" * 40 + "\n")
            
            # Update parameters
            dae_data = json.loads(json.dumps(dae_data_template)) # Deep copy
            for p in dae_data['parameters']:
                if p['name'] == 'g':
                    p['value'] = g_val
                elif p['name'] == 'e':
                    p['value'] = e_val
            
            try:
                # Instantiate Solver
                solver = DAESolver(dae_data, verbose=False)
                
                # Solve Augmented
                t_span = (solver_cfg['start_time'], solver_cfg['stop_time'])
                ncp = solver_cfg['ncp']
                
                aug_sol = solver.solve_augmented(t_span=t_span, ncp=ncp)
                
                # Structure Analysis
                n_seg = len(aug_sol.segments)
                n_ev = len(aug_sol.events)
                
                report.write(f"  Segments: {n_seg}\n")
                report.write(f"  Events: {n_ev}\n")
                
                # Validation Logic
                case_passed = True
                
                # Check 1: Pattern consistency (Segments vs Events)
                # Usually n_seg = n_ev + 1 (last segment ends at t_final)
                # Or n_seg = n_ev (last segment ends exactly at event)
                if n_seg != n_ev + 1:
                    report.write(f"  [INFO] Segment/Event count: {n_seg} segments, {n_ev} events.\n")
                
                # Check 2: Connectivity
                last_t = t_span[0]
                for i, seg in enumerate(aug_sol.segments):
                    
                    if i == 0 and n_ev > 0:
                         ev = aug_sol.events[0]
                         seg0 = aug_sol.segments[0]
                         seg1 = aug_sol.segments[1]
                         
                         report.write("\n  DETAILED PROOF FOR FIRST EVENT:\n")
                         report.write(f"    Segment 0 End Time:   {seg0.t[-1]:.12f}\n")
                         report.write(f"    Event 0 Time:         {ev.t_event:.12f}\n")
                         report.write(f"    Segment 1 Start Time: {seg1.t[0]:.12f}\n")
                         
                         report.write(f"    Segment 0 End State (w-):   {seg0.x[-1]}\n")
                         report.write(f"    Event 0 Pre-State (w-):     {ev.x_pre}\n")
                         report.write(f"    Event 0 Post-State (w+):    {ev.x_post}\n")
                         report.write(f"    Segment 1 Start State (w+): {seg1.x[0]}\n")
                         
                         # Logical checks
                         t_c1 = abs(seg0.t[-1] - ev.t_event) < 1e-9
                         t_c2 = abs(seg1.t[0] - ev.t_event) < 1e-9
                         x_c1 = np.allclose(seg0.x[-1], ev.x_pre, atol=1e-9)
                         x_c2 = np.allclose(seg1.x[0], ev.x_post, atol=1e-9)
                         
                         report.write(f"    [CHECK] Seg0_End_T == Event_T:   {t_c1}\n")
                         report.write(f"    [CHECK] Seg1_Start_T == Event_T: {t_c2}\n")
                         report.write(f"    [CHECK] Seg0_End_W == Event_Pre: {x_c1}\n")
                         report.write(f"    [CHECK] Seg1_Start_W == Event_Post: {x_c2}\n\n")
                    t_start = seg.t[0]
                    t_end = seg.t[-1]
                    
                    # Start connectivity
                    expected_start = last_t
                    if i > 0:
                        expected_start = aug_sol.events[i-1].t_event
                        
                    if abs(t_start - expected_start) > 1e-5:
                        report.write(f"  [FAIL] Segment {i} start time {t_start:.6f} does not match expected {expected_start:.6f}\n")
                        case_passed = False
                    
                    # State continuity after event (reinitialization check)
                    if i > 0:
                        ev_prev = aug_sol.events[i-1]
                        x_start_seg = seg.x[0]
                        x_post_ev = ev_prev.x_post
                        # Compare
                        diff = np.linalg.norm(x_start_seg - x_post_ev)
                        if diff > 1e-5:
                            report.write(f"  [FAIL] Segment {i} start state does not match Event {i-1} post state. Diff={diff:.6e}\n")
                            # Detailed dump for debug
                            report.write(f"         Seg start: {x_start_seg}\n")
                            report.write(f"         Evt post:  {x_post_ev}\n")
                            case_passed = False

                    # End connectivity (check against next event)
                    if i < n_ev:
                        ev_curr = aug_sol.events[i]
                        if abs(t_end - ev_curr.t_event) > 1e-5:
                            report.write(f"  [FAIL] Segment {i} end time {t_end:.6f} does not match Event {i} time {ev_curr.t_event:.6f}\n")
                            case_passed = False
                        
                        # Pre-event state match
                        x_end_seg = seg.x[-1]
                        x_pre_ev = ev_curr.x_pre
                        diff = np.linalg.norm(x_end_seg - x_pre_ev)
                        if diff > 1e-5:
                            report.write(f"  [FAIL] Segment {i} end state does not match Event {i} pre state. Diff={diff:.6e}\n")
                            case_passed = False
                            
                    last_t = t_end

                if case_passed:
                    report.write("  [PASS] Structure verified.\n")
                else:
                    report.write("  [FAIL] Structure verification failed.\n")
                    all_passed = False
                    
            except Exception as e:
                report.write(f"  [ERROR] Exception during simulation: {str(e)}\n")
                all_passed = False
                
            report.write("\n")
            
        report.write("=" * 40 + "\n")
        if all_passed:
            report.write("OVERALL RESULT: PASS\n")
        else:
            report.write("OVERALL RESULT: FAIL\n")
            
    print(f"Verification complete. Report generated at {report_path}")

if __name__ == "__main__":
    verify_dae_output()
