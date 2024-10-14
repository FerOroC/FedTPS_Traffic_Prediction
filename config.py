params = {
    'GRU':{
        'hidden_size':32,
        'num_layers':3,
        'output_size':1,
        'input_size':1,
        'num_nodes':100,
        'lr':1e-3
    },
    'STGCN':{
        'blocks':[1, 64, 64, 64, 64, 64, 32, 32, 128, 128],
        'drop_prob':0.3,
        'n':100,
        'kt':3,
        'lr':1e-3
    },
    'DCRNN':{
        'input_dim':1,
        'output_dim':1,
        'num_nodes':100,
        'hidden_dim':64,
        'num_layers':2,
        'K':3,
        'lr':1e-3
    },
    'GWNET':{
        'drop_prob':0.3,
        'gcn_bool':True,
        'addaptadj':True,
        'n_hid':32,
        'in_dim':1,
        'out_dim':1
    },
    'TAU':{
        'spatio_kernel_enc':3,
        'spatio_kernel_dec':3,
        'model_type':'tau',
        'hid_S':32,
        'hid_T':256,
        'N_T':8,
        'N_S':2,
        'lr':1e-3,
        'drop_path':0.1,
        'in_shape':(12,1,10,10)
    },
    'GATAU':{
        'spatio_kernel_enc': 3,
        'spatio_kernel_dec': 3,
        'hid_S': 16,
        'hid_T': 256,
        'N_T': 8,
        'N_S': 2,
        'lr': 1e-4,
        'drop_path': 0.1,
        'in_shape': (12, 1, 10, 10)
    }
}