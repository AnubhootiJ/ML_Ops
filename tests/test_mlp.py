from bonus import train_mlp

def test_mlp():
    t1_ac = train_mlp(0.01)
    t2_ac = train_mlp(0.01)

    assert t1_ac == t2_ac