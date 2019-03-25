# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 09:40:04 2017

@author: DELL
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 20:06:16 2017

@author: DELL
"""

import numpy as np
import scipy.signal as sc
import sounddevice as sd
import pygame.midi
import pygame.mixer
import pygame
from multiprocessing import Pool


CHUNK = 1024 # longueur d'un chunk
WIDTH = 4
CHANNELS = 1 # nombre de channels
RATE = 44100 # fréquence d'échantillonnage
RECORD_SECONDS = 5
WINSIZE = 0.020
OVERLAP = 0.5
fe = RATE # fréquence d'échantillonnage d'un chunk plus facile à utiliser
N = CHUNK # longueur d'un chunk plus facile à utiliser

T_OLA = int(RATE/100) # longueur d'un incrément OLA (pour zone non pitchée)
h_OLA = np.hanning(2*T_OLA-1) # fenêtre de Hann utilisée pour OLA

def xcorr(x,fe): # algo d'autocorrélation
    """determine les corrélations pour chaque fréquence"""
    
    result =  sc.correlate(x,x,'full')   #on récupère la corrélation
    result = result[int(result.size/2):] #on ne prend qu'une partie car result est symétrique
    return result 

def freq(x,fe):
    """détermine la fréquence du pitch ou retourne -1 si c'est du bruit"""

    ms2 = int(fe/500) # fréquence haute sur au-dessus de laquelle on ne s'intéresse pas à la corrélation (la voix dépasse rarement cette valeur)
    ms20 = int(fe/100) # fréquence basse en dessous de laquelle on ne s'intéresse pas à la corrélation (en dessous, on peut avoir moins de deux périodes par chunk ce qui n'est pas détectable)
    N = len(x) # longueur du signal entrant
    
    corr = xcorr(x,fe)[ms2:ms20]  #on récupère les corrélations dans la plage de fréquence 100Hz - 500 Hz
    argMax = np.argmax(corr)   # indice du maximum de corrélation non replacé sur la longueur totale de 
    
    k = argMax + ms2 # indice du maximum de corrélation replacé au bon endroit
    
    max = corr[argMax]*N/(N-k)    #la fréquence du pitch est celle dont la corrélation est la  meilleure en ayant corrigé le biais
    if max > 0.7:                 #si la corrélation est supérieure au seuil 0.7, on a un pitch
         return fe/(ms2 + argMax) 
    else:                         #sinon c'est un bruit
        return -1 

def PSOLART_chunk(y,h,chunk_a,old_chunk_a,f_piano,old_f_voice,remaining,remaining_h,first_index): 
#fonction qui s'occupe d'implémenter PSOLA avec :
# - chunk_a le chunk d'entrée à analyser
# - old_chunk_a le chunk d'entrée précédent (on peut en avoir besoin pour calculer la premier fenêtre de chunk_a qui peut dépasser à gauche)
# - f_piano la fréquence jouée au piano
# - old_f_voice la fréquence détectée au chunk précédent
# - remaining la partie de la fenêtre de Hann issue du chunk d'analyse précedent à ajouter au chunk de synthèse actuel (fenêtre qui dépasse)
# - first_index le premier index du chunk (utilisé si on vient de OLA et qu'on poursuit OLA) qui nous vient du chunk précédent

    
                                         
    global T_OLA
    global fe
    global h_OLA
    
    T_piano = int(fe/f_piano) # longueur associée à la fréquence de la note jouée au piano (espacement entre deux fenêtres de synthèse dans les zones pitchées)
    f_voice = freq(chunk_a,fe) # fréquence détectée sur chunk_a
   
    if old_f_voice == -1: # si le chunk d'avant était non pitché
        y[N:N+len(remaining)] += chunk_a[:len(remaining)]*remaining# on ajoute le bout de fenêtre issue du chunk précédent
        h[N:N+len(remaining)] += remaining_h
    else: # si le chunk d'avant était pitché

        y[N:N+len(remaining)] += remaining # on ajoute le bout de waveform issu du chunk précédent
        h[N:N+len(remaining)] += remaining_h
    
    if f_voice == -1: # si le chunk est non pitché
        waveform = []
        i = first_index # on initalise le marqueur d'analyse grâce à l'index passé par l'analyse du chunk précédent      
        old_i = N + i - T_OLA   # dernier index dans le chunk précédent
        window = np.concatenate([old_chunk_a[old_i+1:],chunk_a[:i+T_OLA]])*h_OLA # première portion de signal à copier
        y[old_i+1:N+i+T_OLA] += window # on place la première fenêtre dans y en finissant la portion gauche de y qu'on n'avait pas pu envoyer 
        h[old_i+1:N+i+T_OLA] += h_OLA
        for k in range (N):
            if h[k] == 0 :
                h[k] = 1
        y[:N]= y[:N]/h[:N]
        i+=T_OLA # on incrémente i d'une période OLA

        
        while (i + T_OLA < N):
            y[N+i-T_OLA+1:N+i+T_OLA] += chunk_a[i-T_OLA+1:i+T_OLA]*h_OLA # cas agréable
            h[N+i-T_OLA+1:N+i+T_OLA] += h_OLA # cas agréable
            i+=T_OLA # on incrémente i d'une période OLA
            
        remaining = np.zeros(N,np.float32)    
        remaining_h = np.zeros(N,np.float32)
        while (i < N): # on parcourt le chunk d'analyse
            y[N+i-T_OLA+1:] += chunk_a[i-T_OLA+1:]*h_OLA[:N-i+T_OLA-1]
            h[N+i-T_OLA+1:] += h_OLA[:N-i+T_OLA-1]
            remaining[:T_OLA+i-N] += h_OLA[N-i+T_OLA-1:] # calcul du bout de fenêtre de Hann qui dépasse pour l'itération suivante
            remaining_h[:T_OLA+i-N] += h_OLA[N-i+T_OLA-1:] # calcul du bout de fenêtre de Hann qui dépasse pour l'itération suivante         
            i+=T_OLA
        first_index = i-N # calcul du premier index dans l'itération suivante

    else: # si le chunk est pitché
        
        T_voice = int(fe/f_voice) # largeur d'une période du signal d'analyse (détectée)
        pitch_marker = np.argmax(chunk_a[T_voice:-T_voice]) + T_voice # position du maximum (fermeture glottale) autour duquel on va extraire une forme d'onde (on enlève les extrémités pour être sur d'en avoir une entière)
        h_voice = np.hanning(2*T_voice+1)[1:-1] # fenêtre de hann pour PSOLA (de taille liée au pitch détecté)
        waveform = chunk_a[pitch_marker-T_voice+1:pitch_marker+T_voice]*h_voice # forme d'onde extraite
        
            
            
        if old_f_voice == -1 : # si le chunk précédent était non pitché 
        
            if first_index + T_OLA < pitch_marker - T_voice: # cas om l'on continue OLA jusqu'au pitch_marker (forme d'onde sélectionnée "trop à droite")
                i = first_index # on initalise le premier marqueur d'analyse grâce à l'index passé par l'analyse du chunk précédent      
                old_i = N + i - T_OLA  # dernier index dans le chunk précédent
                window = np.concatenate([old_chunk_a[old_i+1:],chunk_a[:i+T_OLA]])*h_OLA # première portion de signal à copier
                y[old_i+1:N+i+T_OLA] += window # on place la première fenêtre dans y en finissant la portion gauche de y qu'on n'avait pas pu envoyer 
                h[old_i+1:N+i+T_OLA] += h_OLA
                for k in range (N) : 
                    if h[k] == 0 :
                        h[k] = 1
                y[:N]=y[:N]/h[:N]
                i+=T_OLA # on incrémente i d'une période OLA   
                
            else:  # cas on l'on commence PSOLA dès le début (forme d'onde sélectionnée "bien à gauche")
                i = pitch_marker # on initialise le premier marqueur d'analyse au niveau du pitch_marker
                left_end = N + i - T_voice # on calcule l'indice du côté gauche de notre première fenêtre qui va dépasser dans l'ancien chunk 
                y[left_end+1:N+i+T_voice] += waveform # on place la première forme d'onde dans y en finissant la portion gauche de y qu'on n'avait pas pu envoyer 
                h[left_end+1:N+i+T_voice] += h_voice
                for k in range (N) : 
                    if h[k] == 0 :
                        h[k] = 1
                y[:N]=y[:N]/h[:N]
                i+=T_piano # on incrémente d'une période de piano
                         
            while i + T_voice < N:
                if i + T_OLA < pitch_marker - T_voice: # si on continue OLA jusqu'au pitch_marker
                    y[N+i-T_OLA+1:N+i+T_OLA] += chunk_a[i-T_OLA+1:i+T_OLA]*h_OLA # cas agréable
                    h[N+i-T_OLA+1:N+i+T_OLA] += h_OLA
                    i+=T_OLA
                else: # si on a déjà commencé PSOLA (on a déjà rencontré le pitch_marker)
                    y[N+i-T_voice+1:N+i+T_voice] += waveform
                    h[N+i-T_voice+1:N+i+T_voice] += h_voice
                    i+=T_piano
                    
            remaining = np.zeros(N,np.float32)
            remaining_h = np.zeros(N,np.float32)
            while i < N:
                y[N+i-T_voice+1:] += waveform[:N-i+T_voice-1]*h_voice[:N-i+T_voice-1]
                h[N+i-T_voice+1:] += h_voice[:N-i+T_voice-1]
                remaining[:T_voice+i-N] += waveform[N-i+T_voice-1:]*h_voice[N-i+T_voice-1:] # calcul du bout de waveform qui dépasse pour l'itération suivante
                remaining_h[:T_voice+i-N] += h_voice[N-i+T_voice-1:]
                i+= T_piano
            first_index = i-N     
            
                    
        else: # si le chunk précédent était pitché
        
            i = first_index # on initalise le marqueur d'analyse grâce à l'index passé par l'analyse du chunk précédent      
            left_end = N + i - T_voice # extrémité gauche de la première fenêtre du chunk actuel qui dépasse dans le chunk précédent
            y[left_end+1:N+i+T_voice] += waveform # on place la première fenêtre dans y en finissant la portion gauche de y qu'on n'avait pas pu envoyer 
            h[left_end+1:N+i+T_voice] += h_voice
            for k in range (N) : 
                    if h[k] == 0 :
                        h[k] = 1
            y[:N]=y[:N]/h[:N]
            i+=T_piano # on incrémente i d'une période OLA
            
            while i + T_voice < N :
                 y[N+i-T_voice+1:N+i+T_voice] += waveform # cas agréable
                 h[N+i-T_voice+1:N+i+T_voice] += h_voice   
                 i+=T_piano # on incrémente i d'une période OLA
            
            remaining = np.zeros(N,np.float32)
            remaining_h = np.zeros(N,np.float32)
            while i < N :               
                y[N+i-T_voice+1:] += waveform[:N-i+T_voice-1]*h_voice[:N-i+T_voice-1]
                h[N+i-T_voice+1:] += h_voice[:N-i+T_voice-1]
                remaining[:T_voice+i-N] += waveform[N-i+T_voice-1:]*h_voice[N-i+T_voice-1:]
                remaining_h[:T_voice+i-N] += h_voice[N-i+T_voice-1:]
                i+=T_piano
                
            first_index = i - N # calcul du premier index dans l'itération suivante
            

    chunk_s = y[:CHUNK]
    y = np.concatenate([y[CHUNK:],np.zeros(CHUNK,np.float32)])
    h = np.concatenate([h[CHUNK:],np.zeros(CHUNK,np.float32)])


    
    return y,h,chunk_s,chunk_a,f_voice,remaining,first_index,remaining_h


def print_device_info():
    pygame.midi.init()
    _print_device_info()
    pygame.midi.quit()
    
    
def _print_device_info():
    for i in range( pygame.midi.get_count() ):
        r = pygame.midi.get_device_info(i)
        (interf, name, input, output, opened) = r

        in_out = ""
        if input:
            in_out = "(input)"
        if output:
            in_out = "(output)"

        print ("%2i: interface :%s:, name :%s:, opened :%s:  %s" %
               (i, interf, name, opened, in_out))
        

#    pygame.mixer.init(44100,-16,1,1024)
#    tab_Sound=[pygame.mixer.Sound(x) for x in liste_sound]

## ouverture du stream d'I/O
sd.default.device = 30 
s = sd.Stream(blocksize = 0, channels = CHANNELS, dtype = np.float32)
s.start()
print("* recording")

## ouverture du MIDI
pygame.init()
pygame.fastevent.init()
event_get = pygame.fastevent.get
event_post = pygame.fastevent.post
pygame.midi.init()
_print_device_info()
i = pygame.midi.Input(2)
pygame.display.set_mode((1,1))
going = True


spot = 0 
F = [[0,False],[0,False],[0,False],[0,False]]
while going:
    
    events = event_get()
    for e in events:
        if e.type in [pygame.midi.MIDIIN]:
            if e.status==144 :
                for j in range (0,4):
                    if F[j][0] == 0:
                        F[j] = [262.63*(2**((e.data1-48)/12)),False]
                        break
            elif e.status == 128:
                note = 262.63*(2**((e.data1-48)/12))
#                print(note)
#                print(F)
                j = 0
                while j<4 and F[j][0] != note:
                    j+=1
                F[j] = [0,True]
                        

    if i.poll():
        midi_events = i.read(20)
    # convert them into pygame events.
        midi_evs = pygame.midi.midis2events(midi_events, i.device_id)
        
        for m_e in midi_evs:
            event_post(m_e)
            
    chunk_a = s.read(CHUNK)[0]
    Y = chunk_a/5
    Y = np.reshape(Y,[len(Y),1])
    chunk_a = np.reshape(chunk_a,len(chunk_a))
    if F[2][0] != 0 :
        if not F[2][1]:
            h2 = np.zeros(2*N,np.float32)
            y2 = np.zeros(2*N,np.float32)
            y2,h2,chunk_s2,old_chunk_a2,old_f_voice2,remaining2,first_index2,remaining_h2 = PSOLART_chunk(y2,h2,chunk_a,np.zeros(CHUNK,np.float32),F[2][0],-1,[],[],0)
            chunk_s2 = np.zeros(CHUNK,np.float32)
            F[2][1] = True
        else:
            y2,h2,chunk_s2,old_chunk_a2,old_f_voice2,remaining2,first_index2,remaining_h2 = PSOLART_chunk(y2,h2,chunk_a,old_chunk_a2,F[2][0],old_f_voice2,remaining2,remaining_h2,first_index2)
        chunk_s2 = chunk_s2/5
        chunk_s2 = np.reshape(chunk_s2,[len(chunk_s2),1])
        Y += chunk_s2
        
    if F[0][0] != 0 :
        if not F[0][1]:
            h0 = np.zeros(2*N,np.float32)
            y0 = np.zeros(2*N,np.float32)
            y0,h0,chunk_s0,old_chunk_a0,old_f_voice0,remaining0,first_index0,remaining_h0 = PSOLART_chunk(y0,h0,chunk_a,np.zeros(CHUNK,np.float32),F[0][0],-1,[],[],0)
            chunk_s0 = np.zeros(CHUNK,np.float32)
            F[0][1] = True
        else:
            y0,h0,chunk_s0,old_chunk_a0,old_f_voice0,remaining0,first_index0,remaining_h0 = PSOLART_chunk(y0,h0,chunk_a,old_chunk_a0,F[0][0],old_f_voice0,remaining0,remaining_h0,first_index0)
        chunk_s0 = chunk_s0/5
        chunk_s0 = np.reshape(chunk_s0,[len(chunk_s0),1])
        Y += chunk_s0     

    if F[1][0] != 0 :
        if not F[1][1]:
            h1 = np.zeros(2*N,np.float32)
            y1 = np.zeros(2*N,np.float32)
            y1,h1,chunk_s1,old_chunk_a1,old_f_voice1,remaining1,first_index1,remaining_h1 = PSOLART_chunk(y1,h1,chunk_a,np.zeros(CHUNK,np.float32),F[1][0],-1,[],[],0)
            chunk_s1 = np.zeros(CHUNK,np.float32)
            F[1][1] = True
        else:
            y1,h1,chunk_s1,old_chunk_a1,old_f_voice1,remaining1,first_index1,remaining_h1 = PSOLART_chunk(y1,h1,chunk_a,old_chunk_a1,F[1][0],old_f_voice1,remaining1,remaining_h1,first_index1)
        chunk_s1 = chunk_s1/5
        chunk_s1 = np.reshape(chunk_s1,[len(chunk_s1),1])
        Y += chunk_s1 
        
    if F[3][0] != 0 :
        if not F[3][1]:
            h3 = np.zeros(2*N,np.float32)
            y3 = np.zeros(2*N,np.float32)
            y3,h3,chunk_s3,old_chunk_a3,old_f_voice3,remaining3,first_index3,remaining_h3 = PSOLART_chunk(y3,h3,chunk_a,np.zeros(CHUNK,np.float32),F[3][0],-1,[],[],0)
            chunk_s3 = np.zeros(CHUNK,np.float32)
            F[0][1] = True
        else:
            y3,h3,chunk_s3,old_chunk_a3,old_f_voice3,remaining3,first_index3,remaining_h3 = PSOLART_chunk(y3,h3,chunk_a,old_chunk_a3,F[3][0],old_f_voice3,remaining3,remaining_h3,first_index3)
        chunk_s3 = chunk_s3/5
        chunk_s3 = np.reshape(chunk_s3,[len(chunk_s3),1])
        Y += chunk_s3         
        
        
    s.write(Y)
    
            
    

del i
pygame.midi.quit()
    
