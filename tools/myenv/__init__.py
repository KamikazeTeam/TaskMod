from gym.envs.registration import register

register(
    id='shipS-v0',
    entry_point='myenv.shipS:ShipS'
)

register(
    id='rts-v0',
    entry_point='myenv.rts:RTS'
)

