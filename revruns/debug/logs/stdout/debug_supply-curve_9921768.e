Traceback (most recent call last):
  File "/shared-projects/rev/modulefiles/alternate_envs/devruns/lib/python3.8/runpy.py", line 194, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/shared-projects/rev/modulefiles/alternate_envs/devruns/lib/python3.8/runpy.py", line 87, in _run_code
    exec(code, run_globals)
  File "/home/twillia2/github/reV/reV/supply_curve/cli_supply_curve.py", line 425, in <module>
    main(obj={})
  File "/shared-projects/rev/modulefiles/alternate_envs/devruns/lib/python3.8/site-packages/click/core.py", line 1130, in __call__
    return self.main(*args, **kwargs)
  File "/shared-projects/rev/modulefiles/alternate_envs/devruns/lib/python3.8/site-packages/click/core.py", line 1055, in main
    rv = self.invoke(ctx)
  File "/shared-projects/rev/modulefiles/alternate_envs/devruns/lib/python3.8/site-packages/click/core.py", line 1657, in invoke
    return _process_result(sub_ctx.command.invoke(sub_ctx))
  File "/shared-projects/rev/modulefiles/alternate_envs/devruns/lib/python3.8/site-packages/click/core.py", line 1635, in invoke
    rv = super().invoke(ctx)
  File "/shared-projects/rev/modulefiles/alternate_envs/devruns/lib/python3.8/site-packages/click/core.py", line 1404, in invoke
    return ctx.invoke(self.callback, **ctx.params)
  File "/shared-projects/rev/modulefiles/alternate_envs/devruns/lib/python3.8/site-packages/click/core.py", line 760, in invoke
    return __callback(*args, **kwargs)
  File "/shared-projects/rev/modulefiles/alternate_envs/devruns/lib/python3.8/site-packages/click/decorators.py", line 26, in new_func
    return f(get_current_context(), *args, **kwargs)
  File "/home/twillia2/github/reV/reV/supply_curve/cli_supply_curve.py", line 255, in direct
    raise e
  File "/home/twillia2/github/reV/reV/supply_curve/cli_supply_curve.py", line 242, in direct
    out = SupplyCurve.full(sc_points, trans_table,
  File "/home/twillia2/github/reV/reV/supply_curve/supply_curve.py", line 1246, in full
    supply_curve = sc.full_sort(fcr, transmission_costs=transmission_costs,
  File "/home/twillia2/github/reV/reV/supply_curve/supply_curve.py", line 1033, in full_sort
    self._check_substation_conns(self._trans_table)
  File "/home/twillia2/github/reV/reV/supply_curve/supply_curve.py", line 368, in _check_substation_conns
    raise SupplyCurveInputError(msg)
reV.utilities.exceptions.SupplyCurveInputError: The following sc_gid (keys) were connected to substations but were not connected to the respective transmission line gids (values) which is required for full SC sort: {2: [1855], 6: [1790, 1794, 1855], 11: [1925], 29: [1725, 1732], 37: [1790, 1794, 1855], 40: [1752, 1753, 1755], 42: [1633], 77: [1853], 78: [1780, 1823], 137: [1632], 166: [1365, 1626], 174: [1064], 241: [2011], 252: [2372, 2560], 267: [1330, 1793, 1804], 295: [2563, 2565], 327: [2349], 366: [1953, 1965], 370: [1835], 397: [2369], 411: [3595, 3616]}
