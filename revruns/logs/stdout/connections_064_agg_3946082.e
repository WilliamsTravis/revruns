Error running reV SC aggregation CLI.
Traceback (most recent call last):
  File "/home/twillia2/.conda-envs/rr/lib/python3.7/site-packages/reV/supply_curve/cli_sc_aggregation.py", line 437, in <module>
    main(obj={})
  File "/home/twillia2/.conda-envs/rr/lib/python3.7/site-packages/click/core.py", line 829, in __call__
    return self.main(*args, **kwargs)
  File "/home/twillia2/.conda-envs/rr/lib/python3.7/site-packages/click/core.py", line 782, in main
    rv = self.invoke(ctx)
  File "/home/twillia2/.conda-envs/rr/lib/python3.7/site-packages/click/core.py", line 1257, in invoke
    sub_ctx = cmd.make_context(cmd_name, args, parent=ctx)
  File "/home/twillia2/.conda-envs/rr/lib/python3.7/site-packages/click/core.py", line 700, in make_context
    self.parse_args(ctx, args)
  File "/home/twillia2/.conda-envs/rr/lib/python3.7/site-packages/click/core.py", line 1212, in parse_args
    rest = Command.parse_args(self, ctx, args)
  File "/home/twillia2/.conda-envs/rr/lib/python3.7/site-packages/click/core.py", line 1048, in parse_args
    value, args = param.handle_parse_result(ctx, opts, args)
  File "/home/twillia2/.conda-envs/rr/lib/python3.7/site-packages/click/core.py", line 1623, in handle_parse_result
    value = self.full_process_value(ctx, value)
  File "/home/twillia2/.conda-envs/rr/lib/python3.7/site-packages/click/core.py", line 1965, in full_process_value
    return Parameter.full_process_value(self, ctx, value)
  File "/home/twillia2/.conda-envs/rr/lib/python3.7/site-packages/click/core.py", line 1589, in full_process_value
    value = self.process_value(ctx, value)
  File "/home/twillia2/.conda-envs/rr/lib/python3.7/site-packages/click/core.py", line 1579, in process_value
    return self.type_cast_value(ctx, value)
  File "/home/twillia2/.conda-envs/rr/lib/python3.7/site-packages/click/core.py", line 1568, in type_cast_value
    return _convert(value, (self.nargs != 1) + bool(self.multiple))
  File "/home/twillia2/.conda-envs/rr/lib/python3.7/site-packages/click/core.py", line 1565, in _convert
    return self.type(value, self, ctx)
  File "/home/twillia2/.conda-envs/rr/lib/python3.7/site-packages/click/types.py", line 46, in __call__
    return self.convert(value, param, ctx)
  File "/home/twillia2/.conda-envs/rr/lib/python3.7/site-packages/rex/utilities/cli_dtypes.py", line 108, in convert
    return [self.dtype(x) for x in list0]
  File "/home/twillia2/.conda-envs/rr/lib/python3.7/site-packages/rex/utilities/cli_dtypes.py", line 108, in <listcomp>
    return [self.dtype(x) for x in list0]
  File "/home/twillia2/.conda-envs/rr/lib/python3.7/site-packages/rex/utilities/cli_dtypes.py", line 125, in dtype
    return float(x)
ValueError: could not convert string to float: 'PLACEHOLDER'
Traceback (most recent call last):
  File "/home/twillia2/.conda-envs/rr/lib/python3.7/runpy.py", line 193, in _run_module_as_main
    "__main__", mod_spec)
  File "/home/twillia2/.conda-envs/rr/lib/python3.7/runpy.py", line 85, in _run_code
    exec(code, run_globals)
  File "/home/twillia2/.conda-envs/rr/lib/python3.7/site-packages/reV/supply_curve/cli_sc_aggregation.py", line 437, in <module>
    main(obj={})
  File "/home/twillia2/.conda-envs/rr/lib/python3.7/site-packages/click/core.py", line 829, in __call__
    return self.main(*args, **kwargs)
  File "/home/twillia2/.conda-envs/rr/lib/python3.7/site-packages/click/core.py", line 782, in main
    rv = self.invoke(ctx)
  File "/home/twillia2/.conda-envs/rr/lib/python3.7/site-packages/click/core.py", line 1257, in invoke
    sub_ctx = cmd.make_context(cmd_name, args, parent=ctx)
  File "/home/twillia2/.conda-envs/rr/lib/python3.7/site-packages/click/core.py", line 700, in make_context
    self.parse_args(ctx, args)
  File "/home/twillia2/.conda-envs/rr/lib/python3.7/site-packages/click/core.py", line 1212, in parse_args
    rest = Command.parse_args(self, ctx, args)
  File "/home/twillia2/.conda-envs/rr/lib/python3.7/site-packages/click/core.py", line 1048, in parse_args
    value, args = param.handle_parse_result(ctx, opts, args)
  File "/home/twillia2/.conda-envs/rr/lib/python3.7/site-packages/click/core.py", line 1623, in handle_parse_result
    value = self.full_process_value(ctx, value)
  File "/home/twillia2/.conda-envs/rr/lib/python3.7/site-packages/click/core.py", line 1965, in full_process_value
    return Parameter.full_process_value(self, ctx, value)
  File "/home/twillia2/.conda-envs/rr/lib/python3.7/site-packages/click/core.py", line 1589, in full_process_value
    value = self.process_value(ctx, value)
  File "/home/twillia2/.conda-envs/rr/lib/python3.7/site-packages/click/core.py", line 1579, in process_value
    return self.type_cast_value(ctx, value)
  File "/home/twillia2/.conda-envs/rr/lib/python3.7/site-packages/click/core.py", line 1568, in type_cast_value
    return _convert(value, (self.nargs != 1) + bool(self.multiple))
  File "/home/twillia2/.conda-envs/rr/lib/python3.7/site-packages/click/core.py", line 1565, in _convert
    return self.type(value, self, ctx)
  File "/home/twillia2/.conda-envs/rr/lib/python3.7/site-packages/click/types.py", line 46, in __call__
    return self.convert(value, param, ctx)
  File "/home/twillia2/.conda-envs/rr/lib/python3.7/site-packages/rex/utilities/cli_dtypes.py", line 108, in convert
    return [self.dtype(x) for x in list0]
  File "/home/twillia2/.conda-envs/rr/lib/python3.7/site-packages/rex/utilities/cli_dtypes.py", line 108, in <listcomp>
    return [self.dtype(x) for x in list0]
  File "/home/twillia2/.conda-envs/rr/lib/python3.7/site-packages/rex/utilities/cli_dtypes.py", line 125, in dtype
    return float(x)
ValueError: could not convert string to float: 'PLACEHOLDER'
