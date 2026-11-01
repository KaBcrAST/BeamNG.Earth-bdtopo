# BeamNG.Earth — copie de travail privée

Copie privée de **[TecnaGamer/BeamNG.Earth](https://github.com/TecnaGamer/BeamNG.Earth)** (licence MIT,
© 2026 TecnaGamer), avec des travaux en cours sur le rendu de la France. Tout l'amont est conservé tel quel,
`LICENSE` compris. Rien n'a été proposé au dépôt d'origine à ce jour.

> GitHub affiche ce fichier à la place de `README.md`, qui est resté celui du projet d'origine.

## ⬇ Télécharger le jeu

### **[BeamNG.Earth-1.6.3-beta-bdtopo.exe](https://github.com/KaBcrAST/BeamNG.Earth-bdtopo/releases/download/v1.6.3-beta-bdtopo/BeamNG.Earth-1.6.3-beta-bdtopo.exe)** (53 Mo)

Téléchargez-le et **double-cliquez-le**. C'est tout : les réglages sont gravés dedans, il n'y a pas de
`.bat` à lancer. La [page de la release](../../releases/tag/v1.6.3-beta-bdtopo) porte aussi la notice et
l'empreinte SHA-256.

> **Vous ne le trouvez pas dans la liste des fichiers ci-dessus ?** C'est normal : un binaire de 53 Mo n'a
> rien à faire dans un historique Git, où il resterait pour toujours. Il vit dans les **Releases**, colonne
> de droite.
>
> Attention à ne pas prendre la release `v1.6.3-beta` : elle a été publiée automatiquement par le workflow
> du projet quand le tag de l'amont a été poussé, et son exécutable est le build **d'origine**, sans aucun
> des travaux sur la France.

Pour le reconstruire depuis les sources :

```
python tools/build_app.py        # écrit dist/BeamNG.Earth-<version>.exe
```

## Les branches

| Branche | Contenu |
|---|---|
| `local-build` | **l'état actuel** : tout réuni, c'est ce dont l'exécutable est issu |
| `france-bdtopo-heights` | les données de l'IGN : hauteurs, matériaux, usages, formes de toit, orientation du faîtage |
| `facade-bay-variety` | les façades : fenêtres par travée, teintes, creux, maçonnerie, murs mitoyens |
| `distance-haze` | le voile atmosphérique au loin |
| `main` | l'amont, intact |

Les trois branches de travail sont écrites pour devenir des contributions au dépôt d'origine : un sujet par
branche, des commits qui expliquent la mesure derrière chaque choix. `local-build` est leur fusion et n'est
pas destinée à être proposée telle quelle — un de ses commits traverse deux sujets et devra être scindé.

## Ce qui a été ajouté, en bref

**Données ouvertes de l'IGN** (Licence Ouverte 2.0) : hauteur, nombre d'étages, matériaux et usage des
bâtiments depuis la **BD TOPO** ; forme des toits, orientation du faîtage et hauteur sous faîtage mesurées
dans le modèle de hauteur **LiDAR HD**. Sur une cellule de Bailleul, la hauteur passe de 13,7 % à 82,9 % des
bâtiments, la forme de toit de 0 % à 92,6 %.

**Façades** : une fenêtre tirée par travée au lieu d'une suite figée, six teintes de brique au lieu de trois,
les fenêtres creusées dans le mur avec leurs retours de tableau, la maçonnerie en vraie brique, et des
fenêtres européennes à deux battants au lieu des fenêtres à guillotine d'origine.

**Toits** : de vraies tuiles de terre cuite au lieu de la texture d'ardoise teintée en orange.

**Performance** : les murs mitoyens, construits deux fois et visibles nulle part, ne sont plus dessinés
(−11,4 % de triangles sur les bâtiments proches, construction 9 % plus rapide).

Le détail de chaque changement, avec ses mesures, est dans les messages de commit et dans la notice jointe
à la Release.

## Ressources externes

Deux textures de [Poly Haven](https://polyhaven.com) en CC0 (brique de mur, tuiles de toit) sont embarquées
sous `mod/beamng_earth/levels/beamng_earth/art/shapes/beamngEarth/`. Les fenêtres européennes viennent d'un
atlas déjà livré avec BeamNG.drive et ne sont pas redistribuées. Voir `CREDITS.md`.
