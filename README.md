# video2vmd multi-pessoa

Captura os movimentos de **duas (ou mais) pessoas** de um vídeo e gera **um `.vmd`
para cada pessoa**, no mesmo formato que o XR Animator exporta. Assim dá para
seguir o fluxo que você já usa no Virt-A-Mate: cada `.vmd` vai para um personagem
pelo plugin **mmd2timeline**.

```
danca.mp4 ──► video2vmd.py ──► danca_pessoa1.vmd  ──► personagem 1 (mmd2timeline)
                          └──► danca_pessoa2.vmd  ──► personagem 2 (mmd2timeline)
                          └──► danca_preview.mp4  (para conferir quem é quem)
```

## Por que não modificar o XR Animator direto?

O XR Animator é todo construído em volta de **um** avatar (a cena, o MMD
exportado, a captura e os filtros assumem uma pessoa só). Ligar um segundo
"boneco" exigiria mexer em boa parte do programa. Esta ferramenta faz a mesma
coisa que importa para você (vídeo → `.vmd`), mas pensada para várias pessoas
desde o início, e usa o mesmo motor de pose (MediaPipe) que o XR Animator.

## Como funciona

1. **Detecção**: um detector de pessoas acha cada pessoa no quadro; cada uma é
   recortada e o MediaPipe Pose (modelo *heavy*) estima o esqueleto 3D dela.
   (O modo multi-pessoa nativo do MediaPipe perde gente com frequência, por isso
   o recorte por pessoa.)
2. **Identidade**: cada pessoa recebe um número fixo. A ferramenta segue a
   posição, o tamanho e a **cor da roupa** de cada um, então quando os dois se
   cruzam ou um some atrás do outro por um instante, os números não trocam.
   *Pessoa 1 = quem está mais à esquerda no início do vídeo.*
3. **Suavização e buracos**: quadros em que alguém sumiu (oclusão) são
   interpolados; o movimento é suavizado com filtro One Euro.
4. **Conversão para MMD**: calcula as rotações dos ossos padrão do MMD
   (`センター, 上半身, 下半身, 首, 頭, 腕, ひじ, 手首, 足, ひざ, 足首`) e grava a
   30 fps. A posição de cada pessoa no palco também é estimada, **mantendo a
   distância entre as duas** (se uma anda até a outra, isso aparece no VaM).

## Instalação (Windows)

1. Instale o **Python 3.10, 3.11 ou 3.12** (marque *"Add Python to PATH"*).
2. Nesta pasta, abra o terminal e rode:
   ```
   pip install -r requirements.txt
   ```
3. Na primeira execução os modelos (~55 MB) são baixados para a pasta `models/`.

## Uso

Jeito mais fácil: **arraste o vídeo para cima de `converter_video.bat`**.

Ou pelo terminal:

```
python video2vmd.py danca.mp4 --preview
```

Gera `danca_pessoa1.vmd`, `danca_pessoa2.vmd` e `danca_preview.mp4`.

### Confira o preview

Abra `danca_preview.mp4`: cada pessoa tem um esqueleto de cor diferente com
`P1`/`P2` em cima e o número do quadro no canto. Se em algum momento os números
trocarem de pessoa, anote o quadro e rode de novo com `--swap`:

```
python video2vmd.py danca.mp4 --swap 450 --preview
```

(a partir do quadro 450, P1 e P2 trocam). Pode repetir `--swap` várias vezes.
A detecção fica em cache (`danca.landmarks.pkl`), então rodar de novo com outras
opções é rápido.

### Opções úteis

| Opção | O que faz |
|---|---|
| `-n 3` | capturar 3 pessoas |
| `--start 5 --end 40` | só um trecho do vídeo (segundos) |
| `--arm-angle 40` | ângulo dos braços na pose de descanso do modelo; se os braços ficarem altos/baixos demais no VaM, ajuste (30–45) |
| `--smooth 2` | mais suave (menos tremido, mais atraso). `0` desliga |
| `--depth 0.5` / `--depth 0` | reduz/desliga o movimento para frente e para trás (a profundidade é a parte menos precisa) |
| `--no-translation` | só rotações, sem mover a センター |
| `--origin each` | cada pessoa começa no próprio ponto zero, em vez de manter a distância entre elas |
| `--no-wrists` | não anima os pulsos (os pontos das mãos do MediaPipe tremem) |
| `--keep-ik` | não desliga o IK das pernas no `.vmd` |
| `--scale 12.5` | unidades MMD por metro |
| `--model full` | modelo mais rápido e menos preciso |

## No Virt-A-Mate

1. Coloque os dois personagens (Person) na cena, **os dois no mesmo ponto** (mesma
   posição/rotação). Com `--origin shared` (padrão) cada `.vmd` já leva o
   deslocamento de cada um, então eles se afastam/aproximam como no vídeo.
2. No mmd2timeline de cada personagem, carregue o seu `.vmd`
   (`_pessoa1` em um, `_pessoa2` no outro) e a mesma música.
3. Se ficarem espelhados ou trocados de lado, use `--origin each` e posicione
   cada personagem manualmente.

## Limitações (honestas)

- Quando os dois estão muito colados (abraço, giro de dança de salão), uma pessoa
  esconde a outra e a pose da escondida fica pior; a ferramenta interpola
  buracos curtos, mas contato preciso (mão na cintura, etc.) costuma precisar
  de um ajuste manual depois.
- Profundidade (frente/trás) vem de uma estimativa a partir do tamanho da pessoa
  na imagem; com câmera se movendo ou zoom, desligue com `--depth 0`.
- Dedos e expressões faciais não são capturados (só corpo).
- Melhores resultados: câmera parada, corpo inteiro visível, as duas pessoas com
  roupas de cores diferentes.

## Para desenvolvedores

```
pip install pytest
python -m pytest tests
```

Estrutura: `mpvmd/extract.py` (detecção), `mpvmd/tracker.py` (identidade),
`mpvmd/filters.py` (suavização), `mpvmd/retarget.py` (landmarks → ossos MMD),
`mpvmd/vmd.py` (escrita do `.vmd`), `video2vmd.py` (linha de comando).
