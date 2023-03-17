
import pathlib
APP_PATH=str(pathlib.Path(__file__).parent.resolve())
CACHE_CONFIG={
    'CACHE_TYPE':'filesystem',
    'CACHE_DIR':APP_PATH+'/dashapp_cache'
}
DATA_FOLDER='data'
DATA_PATH=APP_PATH+'/assets/'+DATA_FOLDER
NOSE_LANDMARK_ID=10
VIDEOFILE_NAME='landmarks_video.mp4'
MEDIAN_FILTERING_LENGTH=7