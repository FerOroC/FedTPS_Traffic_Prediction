import os
import numpy as np
import torch
import scipy.sparse as sp

def load_chengdu_data_new(data_path, synthetic_data_path):

    X = np.load(data_path)
    synth_X = np.load(synthetic_data_path, allow_pickle=True)

    X = X.astype(np.float32)
    X_mean = np.mean(X, axis=(1,2), keepdims=True)
    X_std = np.std(X, axis=(1,2), keepdims=True)

    normalised_X = (X - X_mean) / X_std
    normalised_synth_X = (synth_X - X_mean.mean()) / X_std.mean()

    A = calculate_adjacency_matrix()
    return normalised_X, normalised_synth_X, X_mean, X_std, A

def calculate_adjacency_matrix(n_rows=10, m_columns=10):
    #to improve results, you can change the threshold value of filter in adjacency matrix, or use different distance calculator, or try normalising matrix somehow

    distance_matrix = np.zeros(((n_rows * m_columns), (n_rows * m_columns)), dtype=np.float32)

    for i in range(n_rows * m_columns):
        for j in range(n_rows * m_columns):
            if i//m_columns == j//m_columns:
                distance_matrix[i][j] = np.abs(j-i)
            elif i % m_columns == j % m_columns:
                distance_matrix[i][j] = np.abs((j//m_columns)-(i//m_columns))
            else:
                horizontal = np.abs((i%m_columns)-(j%m_columns))
                vertical = np.abs((i//m_columns)-(j//m_columns))
                distance_matrix[i][j] = np.sqrt(np.square(horizontal)+np.square(vertical))
    distances = distance_matrix[~np.isinf(distance_matrix)].flatten()
    std_distances = distances.std()
    adjacency_matrix = np.exp(-np.square(distance_matrix / std_distances))
    adjacency_matrix[adjacency_matrix<0.05] = 0

    adjacency_matrix = get_normalized_adj(adjacency_matrix)
    
    return adjacency_matrix


def numpy_to_graph(A,type_graph='dgl',node_features=None):

    G = nx.from_numpy_array(A)
    
    if node_features != None:
        for n in G.nodes():
            for k,v in node_features.items():
                G.nodes[n][k] = v[n]
    
    if type_graph == 'nx':
        return G
    
    G = G.to_directed()
    
    if node_features != None:
        node_attrs = list(node_features.keys())
    else:
        node_attrs = []
        
    g = dgl.from_networkx(G, node_attrs=node_attrs, edge_attrs=['weight'])
    return g

def get_normalized_adj(A):
    """
    Returns the degree normalized adjacency matrix.
    """
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave

def data_loader(data):
    len_data = float(data.shape[0])
    train_data_split = 0.6
    val_data_split = 0.2
    test_data_split = 0.2
    inflow_train_dataset = data[:int(len_data*train_data_split),:]
    inflow_val_dataset = data[int(len_data*train_data_split):int((len_data*train_data_split) + (len_data*val_data_split)),:]
    inflow_test_dataset = data[int((len_data*train_data_split) + (len_data*val_data_split)):,:]

    return inflow_train_dataset, inflow_val_dataset, inflow_test_dataset

def data_transform(data, n_his, n_pred, device):
    n_route = data.shape[1]
    length_data = data.shape[0]
    usable_data = length_data - n_his - n_pred
    x = np.zeros([usable_data, 1, n_his, n_route])
    y = np.zeros([usable_data, n_route])

    cnt = 0
    for i in range(length_data - n_his - n_pred):
        head = i
        tail = i + n_his
        x[cnt, :, :, :] = data[head:tail].reshape(1, n_his, n_route)
        y[cnt] = data[tail + n_pred - 1]
        cnt += 1

    return torch.Tensor(x).to(device), torch.Tensor(y).to(device)

def evaluate_model(model, loss, data_iter):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in data_iter:
            y_pred = model(x).view(len(x), -1)
            l = loss(y_pred, y)
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        return l_sum / n

def evaluate_mape(y, y_pred, threshold=2):
    y = y
    y_pred = y_pred
    mask = y > threshold
    if np.sum(mask)!=0:
        mape = np.abs(y[mask] - y_pred[mask])/y[mask]
        return mape
    else:
        return np.nan

def evaluate_metric(model, data_iter, mean, std):
    model.eval()
    with torch.no_grad():
        mae, mape, mse = [], [], []
        mape_homog = []
        for x, y in data_iter:
            y = y.cpu().numpy()
            y_pred = model(x).view(len(x), -1).cpu().numpy()
            y = (y * std) + mean                    #unnormalised y
            y_pred = (y_pred * std) + mean          # unnormalised y_pred
            d = np.abs(y - y_pred)
            mae += d.tolist()
            mape.append(evaluate_mape(y,y_pred))
            mse += (d**2).tolist()
        NMAE = np.array(mae).mean()
        NRMSE = np.sqrt(np.array(mse).mean())
        for item in mape:
            mape_homog.append(item.mean())
        MAPE = np.array(mape_homog).mean()
        return NMAE, NRMSE, MAPE

#------------------Specific for Graph WaveNet-------------------------------

def load_adj(adj_mx, adjtype):
    if adjtype == "scalap":
        adj = [calculate_scaled_laplacian(adj_mx)]
    elif adjtype == "normlap":
        adj = [calculate_normalized_laplacian(adj_mx).astype(np.float32).todense()]
    elif adjtype == "symnadj":
        adj = [sym_adj(adj_mx)]
    elif adjtype == "transition":
        adj = [asym_adj(adj_mx)]
    elif adjtype == "doubletransition":
        adj = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
    elif adjtype == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    else:
        error = 0
        assert error, "adj type not defined"
    return adj

def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    print("adjacency matrix: ", adj.toarray())
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()

def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()

def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian

def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()

def train_model(model, epochs, train_iter, loss, optimizer, scheduler, eval_iter, data_mean, data_std):
    min_val_loss = np.inf
    train_losses = []
    for epoch in range(1, epochs+1):
        l_sum, n = 0.0, 0
        model.train()
        for x, y in train_iter:
            y_pred = model(x).view(len(x), -1)
            l = loss(y_pred, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        scheduler.step()
        val_loss = evaluate_model(model, loss, eval_iter)
        val_nmae, val_nrmse, val_mape = evaluate_metric(model, eval_iter, data_mean, data_std)
        if val_loss < min_val_loss:
            min_val_loss = val_loss
        print(
            "epoch",
            epoch,
            "data mean:",
            data_mean,
            ", train loss:",
            l_sum / n,
            ", validation loss:",
            val_loss,
            ", val NMAE:",
            val_nmae,
            ", val NRMSE:", 
            val_nrmse,
            ", val MAPE:", 
            val_mape*100
        )
        train_losses.append(l_sum / n)
    #need to test every model
    return model, min_val_loss, train_losses
