import EarlyStopping: done, update, update_training

# dummy criterion: stop when loss = training loss
struct Dummy <: StoppingCriterion end

@test update_training(Dummy(), 1.0) === nothing
@test update_training(Dummy(), 1.0, 42.0) == 42.0

update_training(c::Dummy, loss) = (training=loss, loss=nothing)
update_training(c::Dummy, loss, state) = (training=loss, loss=state.loss)
update(c::Dummy, loss, state) = (training=state.training, loss=loss)
done(c::Dummy, state) = state.training == state.loss

stopper  = EarlyStopper(Dummy())
@test !done!(stopper, 1.0, training=true)
@test done!(stopper, 1.0)

true
