import os

from torch import t
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from asyncio import protocols
import pandas as pd 
import numpy as np
from sklearn.preprocessing import (StandardScaler, OrdinalEncoder, LabelEncoder, MinMaxScaler, OneHotEncoder)
from tensorflow.keras.utils import to_categorical
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Normalizer, MaxAbsScaler, RobustScaler, PowerTransformer
import matplotlib.pyplot as plt
import seaborn as sns

train = 'NSL-KDD Data-Subsets\KDDTrain+.txt'
test = 'NSL-KDD Data-Subsets\KDDTest+.txt'
test21 = 'NSL-KDD Data-Subsets\KDDTest-21.txt'
mytest = 'NSL-KDD Data-Subsets\MyTest.txt'

featureV=[
  "duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot",
  "num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root","num_file_creations","num_shells",
  "num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
  "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count", 
  "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
  "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty"
  ]

flagV=['OTH','RSTOS0','SF','SH','RSTO','S2','S1','REJ','S3','RSTR','S0']

protocol_typeV=['tcp','udp','icmp']

serviceV=[
  'http','smtp','finger','domain_u','auth','telnet','ftp','eco_i','ntp_u','ecr_i','other','private','pop_3','ftp_data',
  'rje','time','mtp','link','remote_job','gopher','ssh','name','whois','domain','login','imap4','daytime','ctf','nntp',
  'shell','IRC','nnsp','http_443','exec','printer','efs','courier','uucp','klogin','kshell','echo','discard','systat',
  'supdup','iso_tsap','hostnames','csnet_ns','pop_2','sunrpc','uucp_path','netbios_ns','netbios_ssn','netbios_dgm',
  'sql_net','vmnet','bgp','Z39_50','ldap','netstat','urh_i','X11','urp_i','pm_dump','tftp_u','tim_i','red_i','icmp',
  'http_2784','harvest','aol','http_8001'
  ]
#print(len(serviceV))
binary_attack=[
  'normal','ipsweep', 'nmap', 'portsweep','satan', 'saint', 'mscan','back', 'land', 'neptune', 'pod', 'smurf',
  'teardrop', 'apache2', 'udpstorm', 'processtable','mailbomb','buffer_overflow', 'loadmodule', 'perl', 'rootkit',
  'xterm', 'ps', 'sqlattack','ftp_write', 'guess_passwd', 'imap', 'multihop','phf', 'spy', 'warezclient',
  'warezmaster','snmpgetattack','named', 'xlock', 'xsnoop','sendmail', 'httptunnel', 'worm', 'snmpguess'
  ]

multiclass_attack={ 
  'normal': 'normal',
  'probe': ['ipsweep.', 'nmap.', 'portsweep.','satan.', 'saint.', 'mscan.'],  #сканировании сетевых портов
  'dos': ['back.', 'land.', 'neptune.', 'pod.', 'smurf.','teardrop.', 'apache2.', 'udpstorm.', 'processtable.','mailbomb.'],
  'u2r': ['buffer_overflow.', 'loadmodule.', 'perl.', 'rootkit.','xterm.', 'ps.', 'sqlattack.'],  #получение зарегистрированным пользователем привилегий локального суперпользователя
  'r2l': ['ftp_write.', 'guess_passwd.', 'imap.', 'multihop.','phf.', 'spy.', 'warezclient.', 'warezmaster.','snmpgetattack.','named.', 'xlock.', 'xsnoop.','sendmail.', 'httptunnel.', 'worm.', 'snmpguess.']  #получением доступа незарегистрированного пользователя к компьютеру со стороны удаленного компьютера
  } 
 
train_data = pd.read_csv(train, names=featureV)
test_data = pd.read_csv(test, names=featureV)
test_21 = pd.read_csv(test21,names=featureV)
my_test = pd.read_csv(mytest,names=featureV)

#print(train_data)

train_data = train_data.query("service != 'aol'")
train_data = train_data.query("service != 'harvest'")
train_data = train_data.query("service != 'http_2784'")
train_data = train_data.query("service != 'http_8001'")
train_data = train_data.query("service != 'red_i'")
train_data = train_data.query("service != 'urh_i'")
train_data = train_data.query("service != 'printer'")
train_data = train_data.query("service != 'rje'")

#print(train_data)

test_data = test_data.query("service != 'printer'")
test_data = test_data.query("service != 'rje'")

def preprocessing(data,cls,df):
    data['label'] = data['label'].replace(['normal.','normal'],0) #в столбце label заменяет все значения с normal на 0.

    if cls == 'binary':
        for i in range(len(binary_attack)):
            data['label'] = data['label'].replace(binary_attack[i],1)
    elif cls=='multiclass':
        for i in range(len(multiclass_attack['probe'])):
            data['label'] = data['label'].replace([multiclass_attack['probe'][i],multiclass_attack['probe'][i][:-1]],1)
        for i in range(len(multiclass_attack['dos'])):
            data['label'] = data['label'].replace([multiclass_attack['dos'][i],multiclass_attack['dos'][i][:-1]],2)
        for i in range(len(multiclass_attack['u2r'])):
            data['label'] = data['label'].replace([multiclass_attack['u2r'][i],multiclass_attack['u2r'][i][:-1]],3)
        for i in range(len(multiclass_attack['r2l'])):
            data['label'] = data['label'].replace([multiclass_attack['r2l'][i],multiclass_attack['r2l'][i][:-1]],4)

    y = data['label']                               #содержит значения колонки label из data в диапазоне [0,1...]
    x = data.loc[:,'duration':'hot']                #содержит 300 значений от поля duration до hot(10 значений)+index из data
    
    t = x.protocol_type.copy()                      #значения индекса+протокола
    t = pd.get_dummies(t)                           #кодирует данные фиктивно
    #print('Protocol: ',t.columns.tolist())
    x = x.drop(columns='protocol_type',axis=1)      #удаляет колонку protocol_type
    x = x.join(t)                                   #добавляет колонки кодировки из t (icmp,udp,tcp)

    t1 = x.service.copy()
    t1 = pd.get_dummies(t1)
    #print('Service: ',t1.columns.tolist(),'\nДлина:',len(t1.columns.tolist()))
    #res = [x for x in t1.columns.tolist() + serviceV if x not in t1.columns.tolist() or x not in serviceV] 
    #print(res)
    x = x.drop(columns='service',axis=1)
    x = x.join(t1)

    t2 = x.flag.copy()
    t2 = pd.get_dummies(t2)
    #print('Flag: ',t2.columns.tolist())
    x = x.drop(columns='flag',axis=1)
    x = x.join(t2)

    yt = y.copy()
    yt = pd.get_dummies(yt)

    x = MinMaxScaler(feature_range=(0,1)).fit_transform(x)

    if df=='train':
        return x,yt
    else:
        return x,y

x_train,Y_train = preprocessing(train_data,cls='binary',df='train')
#print(Y_train)
x_test,Y_test = preprocessing(test_data,cls='binary',df='test')

test_21 = test_21.append(my_test, ignore_index=True)        #после обновления pandas необходимо append заменить на concat
#print(test_21[-1:])
x_21_test,y_21_test = preprocessing(test_21,cls='binary',df='test21')


x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))
#print(x_train)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))
x_21_test = np.reshape(x_21_test, (x_21_test.shape[0], x_21_test.shape[1],1))
x_my_test = x_21_test[-1:]
y_my_test = y_21_test[-1:]
