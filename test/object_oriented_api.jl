# from test/criteria.jl:
losses2 = [9.5, 9.3, 10,            # 10.8     0     0     1
           9.3, 9.1, 8.9, 8,        # 11.2     0     0     2
           8.3, 8.4, 9,             # 6.02     12.5  2.08  3
           9.9, 9.5, 10,            # 21.2     25.0  1.18  4
           10.6, 10.4, 11]          # 9.61     37.5  3.90  5

stopper = EarlyStopper(PQ(alpha=3.8, k=2), InvalidValue())

@test !done!(stopper, losses2[1], training=true)
@test !done!(stopper, losses2[2], training=true)
@test !done!(stopper, losses2[3])

@test !done!(stopper, losses2[4], training=true)
@test !done!(stopper, losses2[5], training=true)
@test !done!(stopper, losses2[6], training=true)
@test !done!(stopper, losses2[7])

@test !done!(stopper, losses2[8], training=true)
@test !done!(stopper, losses2[9], training=true)
@test !done!(stopper, losses2[10])

@test !done!(stopper, losses2[11], training=true)
@test !done!(stopper, losses2[12], training=true)
@test !done!(stopper, losses2[13])

@test !done!(stopper, losses2[14], training=true)
@test !done!(stopper, losses2[15], training=true)
@test done!(stopper, losses2[16])

# Test reset
state = stopper.state
reset!(stopper)
@test !EarlyStopping.done(stopper)
reset!(stopper, state)
@test EarlyStopping.done(stopper)

message(stopper) == "Early stop triggered by "*
    "PQ(3.8, 2, 2.220446049250313e-16) stopping criterio
n. "

# verbose case:
stopper = EarlyStopper(InvalidValue(), verbosity=1)

@test_logs (:info, r"training loss: 1.0") done!(stopper, 1.0, training=true)
@test_logs (:info, r"loss: 2.0")  done!(stopper, 2.0)

