import pickle
import sys
sys.path.append('C:\\project\\code')
import setting

def save_pickle_obj(obj, path, name):
	"""save obj as pickle
	Args:
		obj(python obj)
		path(string)
		name(the name of the file without '.pkl')
		"""
	with open(path + name + '.pkl', 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_pickle_obj(path, name):
	with open(path + name + '.pkl', 'rb') as f:
		return pickle.load(f)

