import os, urllib3, multiprocessing, csv, json
from PIL import Image
from io import BytesIO
from tqdm import tqdm

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class Information():
    def __init__(self, data_type):
        if data_type == 'train':
            self.json_in = './data/train.json'
            self.csv_out = './data/train.csv'
            self.img_dir = './data/train/'
        elif data_type == 'val':
            self.json_in = './data/validation.json'
            self.csv_out = './data/val.csv'
            self.img_dir = './data/val/'
        
    def json2dict(self):
        self.id2label = {}
        self.id2url = {}
        
        j = json.load(open(self.json_in))
    
        for annotation in j['annotations']:
            self.id2label[annotation['imageId']] = [int(i) for i in annotation['labelId']]

        for image in j['images']:
            self.id2url[image['imageId']] = image['url']
            
    def dict2csv(self):
        for id_ in self.id2label:
            with open(self.csv_out, 'a') as f:
                writer = csv.writer(f)
                writer.writerow([id_]+self.id2label[id_])
        print('Converted dictionary to {}'.format(self.csv_out))

    def download_img(self, id_url):
        id_, url = id_url
        filename = os.path.join(self.img_dir, '{}.jpg'.format(id_))
        if not os.path.exists(filename):
            try:
                client = urllib3.PoolManager(500)
                response = client.request('GET', url)
                image_data = response.data
                image = Image.open(BytesIO(image_data))
                image_rgb = image.convert('RGB')
                image_rgb.save(filename, format='JPEG', quality=95)
            except:
                print('Warning: Except occured on image id {}'.format(id_))
                return
            
    def dict2img(self):
        pool = multiprocessing.Pool(processes=20)
        
        with tqdm(total=len(self.id2url)) as progress_bar:
            for _ in pool.imap_unordered(self.download_img, list(self.id2url.items())):
                progress_bar.update(1)
