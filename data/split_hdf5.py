import h5py
import sys

filename = sys.argv[1]
rate = float(sys.argv[2])

f_s = h5py.File(filename + '_s', 'w')
f_l = h5py.File(filename + '_l', 'w')

f = h5py.File(filename, 'r')
sp = int(len(f["input_ids"]) * rate)

for key in f.keys():
    f_s.create_dataset(key, data=f[key][0:sp], dtype=f[key].dtype, compression='gzip')
    f_l.create_dataset(key, data=f[key][sp:], dtype=f[key].dtype, compression='gzip')


f_s.flush()
f_l.flush()

f.close()
f_s.close()
f_l.close()