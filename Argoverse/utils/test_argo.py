from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader

root_dir = 'C://users/cathx/repos/argoverse-api/'
data_dir = root_dir + 'train/'

argoverse_loader = ArgoverseTrackingLoader(data_dir)

print('\n Total number of logs:',len(argoverse_loader))
