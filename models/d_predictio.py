import pickle as pkl

user_data="user data"


def Diabities():
    loaded_model = pkl.load(open('Diabities_model.sav', 'rb'))
    result = loaded_model.predict(user_data)
    return result

def hyperten():
    loaded_model = pkl.load(open('Hypertension.sav', 'rb'))
    result = loaded_model.predict(user_data)
    return result

def stroke():
    loaded_model = pkl.load(open('stroke.sav', 'rb'))
    result = loaded_model.predict(user_data)
    return(result)


