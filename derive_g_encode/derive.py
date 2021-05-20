import pickle
def read_pickle(name):
    with open(name, 'rb') as f:
        data = pickle.load(f)
    return data

def write_pickle(data, name):
    with open(name, 'wb') as f:
        pickle.dump(data, f)

# src_domain = ['ND','VT','NH','ME', 'WA','MT','SD','MN','WI','MI','NY','MA','OR','ID','WY','NE','IA','IL', 'IN','OH', 'PA', 'NJ','CT','RI']
# tgt_domain = ['GA', 'OK', 'NC', 'SC', 'LA', 'KY', 'UT', 'MS', 'FL', 'MO', 'MD', 'DE', 'CO', 'CA', 'TN', 'TX', 'KS', 'AZ', 'NV', 'AL', 'VA', 'AR', 'WV', 'NM']

# all_domain = src_domain + tgt_domain

read_file = '19_pred.pkl' # '3_pred.pkl'

info = read_pickle(read_file)
z = info['z']
# print(z.shape)
g_encode = dict()
for i in range(30):
    g_encode[str(i)] = z[i]

write_pickle(g_encode, "g_encode.pkl")
print("success!")
