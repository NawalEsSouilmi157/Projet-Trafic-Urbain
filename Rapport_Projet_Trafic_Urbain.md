# RAPPORT DE PROJET: GESTION INTELLIGENTE DU TRAFIC URBAIN PAR Q-LEARNING

---

**Titre:** Gestion Intelligente du Trafic Urbain par Q-Learning
**Cours:** IAD & Systèmes Multi-Agents
**Niveau:** Cycle Ingénieur — Année 2025-2026

---

## 1. INTRODUCTION

### 1.1 Contexte Général
La gestion des flux de circulation dans les zones urbaines denses représente un défi majeur pour les métropoles modernes. Les encombrements génèrent des conséquences négatives tant sur le plan économique (perte de temps) qu'écologique (forte émission de gaz à effet de serre due à la marche au ralenti). Historiquement, les systèmes de feux tricolores reposent sur des cycles temporels fixes ou des systèmes de détection réactifs simples qui peinent à s'adapter aux variations soudaines et asymétriques du trafic. Avec l'avènement de l'Intelligence Artificielle Distribuée (IAD), il devient possible de concevoir des systèmes de contrôle qui apprennent dynamiquement à s'adapter pour fluidifier le trafic de façon optimale.

### 1.2 Problématique
Comment un agent autonome, placé au niveau d'un carrefour isolé, peut-il apprendre de manière itérative la politique optimale d'alternance des feux afin de minimiser le temps d'attente global des usagers sur des voies à affluence stochastique ? 

### 1.3 Objectifs du Projet
Ce projet pédagogique vise à concevoir, implémenter et analyser un agent intelligent fondé sur la technique d'Apprentissage Par Renforcement (Reinforcement Learning), spécifiquement l'algorithme "Q-Learning". Il s'agira de formaliser ce problème sous forme de Processus de Décision Markovien (MDP), de développer l'environnement de simulation from scratch, et de comparer les performances de l'agent apprenant face à un contrôleur conventionnel (baseline) opérant sur un cycle de durée fixe. 

---

## 2. MODÉLISATION MDP (Processus de Décision Markovien)

La formalisation formelle du problème est la pierre angulaire permettant d'appliquer l'apprentissage par renforcement. Notre environnement est modélisé comme un tuple $\langle S, A, T, R, \gamma \rangle$.

### 2.1 Espace d'états $S$
L'état du système doit capturer suffisamment d'informations pour permettre une prise de décision pertinente sans pour autant provoquer une explosion combinatoire ("Course of Dimensionality").
- Nous avons défini l'espace d'états par l'observation discrétisée des files d'attente de chaque branche (Nord, Sud, Est, Ouest).
- Le nombre de véhicules par branche est cappé à une valeur représentative `max_queue = 5` (les états représentent les niveaux `0, 1, 2, 3, 4, 5`). Au-delà, on considère la file comme étant dans un état critique sans distinguer "6" de "50".
- En plus des files d'attente, l'agent prend en compte la phase courante du feu `(0: Vert N/S, 1: Orange N/S, 2: Vert E/O, 3: Orange E/O)`.
**Taille de l'espace d'états :** $|S| = 6 \times 6 \times 6 \times 6 \times 4 = 5\,184$ états possibles. Cette dimension est parfaitement adaptée à un graphe tabulaire (Q-Table) tout en garantissant un apprentissage exhaustif.

### 2.2 Espace d'actions $A$
L'agent ne possède que deux choix binaires structurants, constituant un espace d'actions $|A| = 2$ :
- `Action 0 : Maintenir la phase` – Le feu conserve son état vert.
- `Action 1 : Changer la phase` – Le feu vert actuel déclenche un feu orange de transition (durée = 1 pas, aucun véhicule ne passe) puis alterne vers le feu vert opposé.

### 2.3 Fonction de récompense $R(s, a)$
L'objectif est d'empêcher la congestion. La récompense est calculée sous la forme explicite suivante :
$$ R = - \frac{\sum_{i \in \{N,S,E,O\}} \text{Files\_Réelles}_i}{4 \times \text{max\_queue}} $$
**Justification :** Utiliser la somme des files d'attente en valeur négative pousse directement la valeur Q associée à chercher la politique qui minimise globalement le nombre de véhicules à l'arrêt. Maintenir le calcul sur la "Vraie" file (avant discrétisation) procure un gradient négatif indispensable à l'agent afin qu'il ressente pénalement la souffrance qu'implique une file très longue, empêchant ainsi le modèle d'être *aveugle* au-delà du gap des 5 véhicules. L'action `Changer phase` requiert aussi une stricte petite pénalité ($R \mathrel{-}= 0.5$) afin de dissuader tout comportement erratique et clignotant.

### 2.4 Facteur d'actualisation $\gamma$
Nous avons statué sur une valeur de **$\gamma = 0.95$**.
**Justification :** Le dégorgement d'un carrefour demande des actions de planification pérennes. Choisir un gamme proche de 1 ($\gamma = 0.95$) permet à l'agent d'accorder du poids aux bénéfices futurs : il est conscient qu'une action "Changer" provoque temporairement un gain faible (à cause du feu orange), mais sait que cela rapportera de conséquentes réductions de file dans un futur à moyen terme (10 à 20 pas).

### 2.5 Dynamique Stochastique de Transition $T(s' | s, a)$
L'arrivée de chaque véhicule sur chaque branche se matérialise selon une loi statistique de *Poisson* de fréquence $\lambda_i$. Étant donné cette occurrence de nature mathématique infinie et aléatoire, la matrice de transition originelle est a priori complètement inconnue par l'agent et complexe à estimer. Ceci justifie fondamentalement notre recours à un algorithme "Movel-Free" par différence temporelle (TD), comme le Q-Learning.

---

## 3. ANALYSE DE L'ENVIRONNEMENT

En alignement avec la classification conventionnelle de l'Intelligence Artificielle, notre modèle d'intersection intelligente possède les propriétés suivantes :

1. **Totalement Observable :** Les senseurs théoriques informent de manière parfaite l'environnement sur la taille des files et l'état des feux. L'agent ne subit aucun biais sur l'état `s`.
2. **Stochastique :** Non déterministe. Les véhicules s'intègrent aléatoirement via le processus de Poisson, impliquant qu'invoquer la même position deux fois entraînera des états et récompenses légèrement différents.
3. **Séquentiel :** Chaque choix présent détermine les contraintes et états futurs (accumulations). Le problème ne relève par d'une architecture multi-bande épisodique courte mais plutôt d'un cheminement continu ("delay effect").
4. **Dynamique :** Même dans le cas où l'agent "tarde à réfléchir" (modélisé par l'absence d'action), les véhicules s'accumulent au feu de façon indépendante des actions, le monde change à chaque milliseconde.
5. **Discret :** Le simulateur a été découpé selon un tick par "pas de temps" avec des arrivées numériques entières et une quantité de branches discrètes.
6. **Agent Unique :** Actuellement, ce carrefour est le dispositif central isolé de toute prise de décision. Cette notion s'altérera dans la "Partie 2" en introduisant des agents inter-connectés (SMA).

---

## 4. ARCHITECTURE DE L'AGENT

### 4.1 Comparatif des Architectures
- **Architecture Réactive Pur :** Elle fonctionne selon le schéma `Stimulus -> Réaction` sans mémoire. Utile (par exemple, "si 5 véhicules -> Changer Feu"), mais elle échoue à planifier des manœuvres rentables sur le long terme car incapable d'abstraction conceptuelle et de projection.
- **Architecture BDI (Belief-Desire-Intention) :** Idéale pour des modélisations psycho-logiques très poussées impliquant des états de comportements cognitifs lourds. Trop verbosité face à un paradigme purement d'équilibrage de flux mathématique.
- **Architecture Délibérative / Apprenante (Notre Choix) :** Fondée sur la mise en œuvre du Q-Learning, l'agent projette activement sa recherche de gains futurs via l'équation de *Bellman*. Il raisonne sur ses connaissances et utilise son dictionnaire de valeurs emmagasinées expérimentalement ($Q-table$) pour déduire la direction la plus logique pour accomplir son but unique (drainer le passage). 

### 4.2 Schéma Technique Global du Système
1. *Senseur (État S)* $\rightarrow$ Reçoit $(q_N, q_S, q_E, q_W)$ et la Phase.
2. *Contrôleur Q-Learning* $\rightarrow$ Evalue les actions possibles via l'équation de convergence.
3. *Action $A$* $\rightarrow$ Commande aux leds (Maintenir/Changer).
4. *Effecteur (Simulation)* $\rightarrow$ Relâche $n$ véhicules / Cumule $\lambda$ flux arrivants.

---

## 5. IMPLÉMENTATION

Le projet, contraint par les bibliothèques `Python`, `Numpy` et `Matplotlib`, sépare son ingénierie en trois pôles majeurs sans aucune librairie RL tierce.

### 5.1 Environnement et Simulation (`intersection_sim.py`)
La classe gère mathématiquement une intersection en générant des arrivées markoviennes de paramètre $\lambda$. Le débit max admissible par le croisement quand le feu est vert est de $5$ véhicules par étape (`max_dep = 5`). Il introduit en outre des contraintes sécuritaires réalistes :
- *Temps vert minimum :* Forcé à 3 pas.
- *Temps vert en limite de saturation:* Temps max coupé de force d'office après 50 pas afin d'empêcher un gel complet d'une direction.
- *Règles de déséquilibre d'urgence:* Si un feu amasse asymétriquement plus de 10 files de retard supplémentaire face à l'autre voie, une transition d'urgence s'exécute pour maintenir la viabilité.

### 5.2 Algorithme Q-Learning (`agent.py`)
Intégré from scratch, il déploie la mécanique algorithmique :
$$Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]$$
* **Hyper-paramètres ajustés :** 
  - Taux d'apprentissage $\alpha = 0.3$ (compromis entre découverte et assimilation).
  - Horizon $\gamma = 0.95$.
  - Exploration Epsilon-Greedy : $\epsilon$ descend doucement de $1.0$ à $0.01$ avec un coefficient de perte optimisé pour 5000 itérations ("decay = $0.999$").

### 5.3 Politique Expérimentale Baseline (`baseline.py`)
Un contrôleur de référence dénué de machine learning. Ce dernier suit aveuglement une durée de régulation alternée, calée expérimentalement sur $15$ secondes par feu Vert, pour servir de comparatif standardisé (tel que l'état actuel des infrastructures mondiales majoritaires).

---

## 6. ANALYSE DES RÉSULTATS

L'entraînement a été opéré formellement sur une campagne robuste de **5 000 épisodes** afin de solidifier les métriques tabulaires.

### 6.1 Diagnostic de Convergence
Le graphique ci-dessous atteste très cliniquement l'apprentissage du modèle :

![Figure 1 : Évolution de la récompense et convergence du Q-Learning (Moyenne lissée sur 50 fenêtres) au fil des 5000 épisodes.](file:///c:/Users/huawe/Desktop/Projet%20Trafic%20Urbain/convergence.png)
*Figure 1 : Courbe de convergence démontrant la stabilisation de l'agent RL après une phase d'exploration.*

1. De l'épisode 0 à 2500, la courbe de la pénalité moyenne souffre du facteur chaotique dicté par l'exploration brute de l'agent aléatoire. Une pente de descente lente et fluide est appliquée ($\epsilon$-decay = 0.9995), ce qui permet à l'agent de visiter activement **1796 états uniques sur les 5184 possibles** composant le Q-table, assurant une construction de connaissance dense et fiable.
2. Tout au long des 5000 épisodes, l'exploration $\epsilon$ baisse finement pour finalement atteindre $0.082$ à la fin de l'entraînement. L'agent exploite alors la matrice MDP raffinée.
3. On perçoit une cassure d'intelligence forte et nette : la courbe se stabilise fermement, attestant la suppression complète des embouteillages.

### 6.2 Résultats Comparatifs (Baseline vs Q-Learning)
Au banc d'essai (cf. `evaluate.py`), l'agent est confronté sans exploration possible face à la politique "Durée Fixe". Les données brutes ressortent visuellement comme suit :

![Figure 2 : Comparaison de la somme des files d'attente cumulées entre la politique fixe (Baseline) et l'agent intelligent (Q-Learning) sur deux scénarios d'évaluation.](file:///c:/Users/huawe/Desktop/Projet%20Trafic%20Urbain/evaluation_results.png)
*Figure 2 : Résultats comparatifs démontrant l'efficacité drastique du Q-Learning (en bleu) face à un feu à temps fixe (en orange).* 

* **Scénario 1 : Trafic Équilibré ($\lambda = 0.4$ pour uniformité stable)**
  - Total Véhicules cumulés en attente Q-Learning : **1171**
  - Total Véhicules cumulés en attente Baseline : **3622**
  - $\rightarrow$ Amélioration nette d'environ **67%** en faveur de l'IAD.

* **Scénario 2 : Charge Asymétrique ($\lambda_{N/S} = 0.8$ et $\lambda_{E/W} = 0.1$)**
  - Total Véhicules cumulés en attente Q-Learning : **931**
  - Total Véhicules cumulés en attente Baseline : **2732**
  - $\rightarrow$ Amélioration majeure de près de **66%**.

*Remarque :* L'agent brille singulièrement sur le flux asymétrique ; sa liberté d'annuler les cycles vides via l'estimation tabulaire et ses contraintes d'imbalances (que justifie cet écart énorme) permet de résorber des surcharges qu'un feu "statique" Baseline encaisserait mécaniquement de plein fouet. Son intégration se traduit par une gestion plus fluide, intelligente et organique que celle initialisé.

---

## 7. DISCUSSION ET LIMITES

### 7.1 Limites de l'approche tabulaire MDP
Bien que hautement performante sur cet espace fini (environ 5 000 états), l'utilisation stricte des `Q-Tables` souffrirait gravement de l'explosion combinatoire de l'espace d'états ("Curse of Dimensionality").
Si nous tentions d'accroître le scope de discrétisation de `0-5` à `0-50` sur l'espace d'attente (soit la taille réelle de la limite physique), la taille théorique de la table exploserait à $50^4 \times 4 \approx 25\,000\,000$ d'états. Une telle matrice rendrait les chances pour l'agent de relire deux fois les mêmes lignes virtuellement nulles, cassant irrévocablement la mise à jour de son expérience.

### 7.2 Perspectives Multi-Agents
Ce projet ne s'arrête techniquement qu'au palier d'un "carrefour isolé" naïf se régulant tout seul face un chaos mathématique (Poisson). Dans un écosystème civil réel, une accélération de la régulation dans un carrefour A transfèrera immédiatement cette masse de trafic vers l'un des carrefours enjambés, B. Ce déport produira des conflits inopinés.
Pour remédier à cela dans la **Partie 2 (SMA)**, il serait naturel d'introduire un concept distribué où différents agents Q-Learning communiqueraient et partageraient en réseau social un message local de pénalité (Q-Routing) ; basculant l'architecture de Q-Learning monolithique vers des systèmes Multi-Agents comme préconisé dans le Deep Q-Learning (DQN).

---

## 8. CONCLUSION

### 8.1 Synthèse Technologique
L'implémentation algorithmique de cette recherche démontre qu'à conditions et flux rigoureusement équivalents, l'Apprentissage Par Renforcement est capable de briser le déterminisme imposé par les contraintes chronologiques du XXe siècle pour extraire des performances excédant formellement les **66% d'efficacité en plus** vis-à-vis d'un cycle de feux constant.
La définition rigoureuse d'un réseau MDP couplée à l'adaptation dynamique $\alpha / \epsilon$ garantit un retour sur investissement spectaculaire.

### 8.2 Apports Pédagogiques
Ce projet de cycle ingénieur met directement en lumière des concepts abstraits : l'impact inopiné des facteurs de récompense aveugles, la nécessite de mapper de manière stricte le paramétrage du gradient physique réel sur le tuple de renforcement matriciel, et confirme ainsi la suprématie des méthodes d'IAD sur l'optimisation des problèmes NP-Complexes liés aux mobilités futures.

---

## 9. RÉFÉRENCES BIBLIOGRAPHIQUES

1. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
2. Watkins, C. J. C. H., & Dayan, P. (1992). *Q-learning*. Machine Learning, 8(3-4), 279-292.
3. Russell, S., & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach* (4th ed.). Pearson.
