from collections import defaultdict

import numpy as np



from keras.preprocessing import text, sequence


def smile_w2v_pad(smile, maxlen_,victor_size):

    # print("smile11111111111111111=", smile)
    #keras API
    tokenizer = text.Tokenizer(num_words=100, lower=False, filters="　")
    tokenizer.fit_on_texts(smile)
    # print("smile1=",smile[0])
    smile_ = sequence.pad_sequences(tokenizer.texts_to_sequences(smile), maxlen=maxlen_) #得到词索引,然后padding成同样大小的list
    # print("smile2=", smile_[0])
    word_index = tokenizer.word_index
    print("word_index=", word_index)
    nb_words = len(word_index)
    # print("nb_words=", nb_words) #14
    smileVec_model = {}
    with open("./Atom.vec", encoding='utf8') as f:
        for line in f:
            values = line.rstrip().rsplit(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            smileVec_model[word] = coefs
    print("add smileVec finished....")


    count=0
    embedding_matrix = np.zeros((nb_words + 1, victor_size))
    for word, i in word_index.items():
        embedding_glove_vector=smileVec_model[word] if word in smileVec_model else None
        if embedding_glove_vector is not None:
            count += 1
            embedding_matrix[i] = embedding_glove_vector
        else:
            print("NONONONONONONONONONONONONONONONONo")
            print("word=", word)
            unk_vec = np.random.random(victor_size) * 0.5
            unk_vec = unk_vec - unk_vec.mean()
            embedding_matrix[i] = unk_vec

    del smileVec_model
    print(embedding_matrix.shape) #(15,100)

    return smile_, word_index, embedding_matrix

if __name__ == "__main__":

    # DATASET, radius, ngram = 'human', 2, 2
    radius, ngram = 2, 2

    atom_dict = defaultdict(lambda: len(atom_dict))
    bond_dict = defaultdict(lambda: len(bond_dict))
    fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
    edge_dict = defaultdict(lambda: len(edge_dict))
    word_dict = defaultdict(lambda: len(word_dict))
    # print("atom_dict=",atom_dict) atom_dict= defaultdict(<function <lambda> at 0x7fd342cff9d8>, {})
    Smiles, compounds, adjacencies, proteins, interactions = '', [], [], [], []

    solute = "F S ( = O ) ( = O ) O c c c c ( C ( c c c c ( O S ( = O ) ( = O ) O c c c c ( C ( c c c c ( O S ( = O ) ( = O ) O c c c c ( C ( c c c c ( O S ( = O ) ( = O ) F ) c c ) c c c c ( O S ( = O ) ( = O ) F ) c c ) c c ) c c ) c c c c ( O S ( = O ) ( = O ) O c c c c ( C ( c c c c ( O S ( = O ) ( = O ) F ) c c ) c 1 c c c ( O S ( = O ) ( = O ) F ) c c ) c c ) c c ) c c ) c c ) c c c c ( O S ( = O ) ( = O ) F ) c c ) c c "
    ACE_solvent = "C C # N"
    NMF_solvent = "O = C N C"
    wat_solvent = "O"
    DMF_solvent = "O = C N ( C ) C"
    meth_solvent = "C c c c c c c"

    smile = []
    smile.append(solute)
    smile.append(ACE_solvent)
    smile.append(NMF_solvent)
    smile.append(wat_solvent)
    smile.append(DMF_solvent)
    smile.append(meth_solvent)

    smile_, smi_word_index, smi_embedding_matrix = smile_w2v_pad(smile, 100, 100)
    print("smile_=", smile_)
    np.save("./smile.npy", smile_)
