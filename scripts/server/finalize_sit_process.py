"""Reuse the paired gate unchanged for the independently trained process v2."""
import finalize_sit_transfer as evaluator
evaluator.task='Mjlab-SitStandProcess-Flat-MicroDuck-HD1910'
evaluator.experiment='hd1910_sitstand_process_v2'
evaluator.run_suffix='process1000'
evaluator.extra_conditions={'gain-low':['--actuator-kp-scale','.8'],
                            'gain-high':['--actuator-kp-scale','1.2']}
evaluator.main()
