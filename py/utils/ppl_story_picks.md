# Perplexity degradation — qualitative story picks (J=5 / J=6 / J=8)

From `cc_prompt_compare` (15 prompts, T=0, C=16, λ=1). Full log: `final_results_runs/cc_prompt_compare/generations.md`.

## Aggregate metrics

| Config | WikiText PPL | Prompt-gen PPL | TPS | Δ vs J=8 (WT) |
|--------|-------------|----------------|-----|---------------|
| J=8, λ=1 | 6.9649 | 1.2596 | 8.08 | — |
| J=6, λ=1 | 7.0593 | 1.2518 | 9.43 | +1.4% |
| J=5, λ=1 | 7.2027 | 1.2869 | 10.23 | +3.4% |

**Takeaway:** J=8 is best on WikiText (pins all 8 router experts in the CC mask). J=5→J=8 spans only ~3.4% WikiText PPL — qualitatively silent on most factual completions (#7, #8, #14). Occasional per-prompt divergence (#1 France) shows aggregate PPL doesn't rank every output.

---

## Prompt 7: A — ~3% PPL span, nearly identical text (J=5 ≈ J=6 ≈ J=8)

**Stem:** `Photosynthesis converts sunlight into chemical energy in plants. The primary pigment responsible is`

*Photosynthesis/chlorophyll: all three open the same way despite J=5→J=8 WikiText gap.*

### J=8 (WikiText PPL 6.96, prompt PPL 1.26)

```text
chlorophyll, which absorbs light in the blue and red wavelengths, reflecting green light, giving plants their green color. The process occurs in two stages: the light-dependent reactions and the Calvin cycle (light-independent reactions). The light-dependent reactions take place in the thylakoid membranes of the chloroplasts, where light energy is converted into chemical energy in the form of ATP …
```

### J=6 (WikiText PPL 7.06, prompt PPL 1.25)

```text
chlorophyll, which is found in the thylakoid membranes of the chloroplasts. The process of photosynthesis can be divided into two main stages: the light-dependent reactions and the light-independent reactions (Calvin cycle). The light-dependent reactions occur in the thylakoid membranes and involve the absorption of light by chlorophyll, which excites electrons and initiates a series of electron t…
```

### J=5 (WikiText PPL 7.20, prompt PPL 1.29)

```text
chlorophyll, which is found in the thylakoid membranes of the chloroplasts. The process of photosynthesis can be divided into two main stages: the light-dependent reactions and the light-independent reactions (Calvin cycle). The light-dependent reactions occur in the thylakoid membranes and involve the absorption of light by chlorophyll, which excites electrons and leads to the production of ATP a…
```

---

## Prompt 8: A — factual continuation holds across all J

**Stem:** `DNA stores genetic information using four bases: adenine, thymine, guanine, and`

*DNA→cytosine: small aggregate PPL differences don't move the answer.*

### J=8 (WikiText PPL 6.96, prompt PPL 1.26)

```text
cytosine. The sequence of these bases determines the genetic code, which is then used to build proteins. The human genome has about 3 billion base pairs, and the entire sequence of these bases is called the genome. The genome is divided into 23 pairs of chromosomes, with each chromosome containing a long chain of DNA. The DNA is organized into genes, which are segments of DNA that code for specifi…
```

### J=6 (WikiText PPL 7.06, prompt PPL 1.25)

```text
cytosine. The sequence of these bases in DNA determines the genetic code. In a DNA molecule, the number of adenine and thymine is equal, and the number of guanine and cytosine is equal. This is known as Chargaff's rule. If a DNA molecule has 20% adenine, what is the percentage of guanine? To solve this problem, we need to use Chargaff's rule, which states that in DNA, the amount of adenine (A) is …
```

### J=5 (WikiText PPL 7.20, prompt PPL 1.29)

```text
cytosine. The sequence of these bases in DNA determines the genetic code. The DNA sequence is read in triplets, called codons, which correspond to specific amino acids. The genetic code is a set of rules that maps codons to amino acids. The genetic code is nearly universal, with a few exceptions. The genetic code is a triplet code, meaning that each codon is three nucleotides long. The genetic cod…
```

---

## Prompt 14: A — all three get 60 mph

**Stem:** `If a car travels 120 miles in 2 hours, its average speed is`

*Arithmetic stem: correct fact at every J, then shared drift into word problems.*

### J=8 (WikiText PPL 6.96, prompt PPL 1.26)

```text
60 mph. But if the car travels at 60 mph for 1 hour and then immediately decelerates to 30 mph for another hour, what is the average speed for the whole trip? To find the average speed for the whole trip, we need to calculate the total distance traveled and divide it by the total time taken. First, let's calculate the distance traveled during each part of the trip: 1. When the car is traveling at …
```

### J=6 (WikiText PPL 7.06, prompt PPL 1.25)

```text
60 mph. But if the car travels at 60 mph for 1 hour and then immediately decelerates to 0 mph in 10 minutes, what is the total distance traveled? Also, is the average speed the same as the average of the speeds? To find the total distance traveled, we need to calculate the distance covered during the two different phases of the trip. First, the car travels at 60 mph for 1 hour. The distance for th…
```

### J=5 (WikiText PPL 7.20, prompt PPL 1.29)

```text
60 mph. But if the car travels at 60 mph for 2 hours, how far does it go? Also, what is the difference between average speed and average velocity? A car is moving at 60 mph and then comes to a stop in 0.5 seconds. What is its acceleration? A car is moving at 60 mph and then accelerates at 3 m/s² for 10 seconds. What is its final velocity? A car is moving at 60 mph and then accelerates at 3 m/s² fo…
```

---

## Prompt 5: B — J=8/J=6 good, J=5 exam junk

**Stem:** `The largest planet in our solar system is`

*Shows J=5 degradation can appear as format collapse, not just wrong facts.*

### J=8 (WikiText PPL 6.96, prompt PPL 1.26)

```text
Jupiter. The mass of Jupiter is 1.90 × 10^27 kg, and its radius is 7.14 × 10^7 m. What is the acceleration due to gravity on the surface of Jupiter? (Assume G = 6.67 × 10^-11 N·m²/kg²) Okay, so I need to find the acceleration due to gravity on the surface of Jupiter. Hmm, I remember that the formula for gravitational acceleration is something like g = GM/r², right? Let me make sure. Yeah, Newton's…
```

### J=6 (WikiText PPL 7.06, prompt PPL 1.25)

```text
Jupiter. The mass of Jupiter is about 318 times that of Earth, and its radius is about 11 times that of Earth. How does the density of Jupiter compare to the density of Earth? (Assume that the mass and radius of Jupiter are 318 and 11 times those of Earth, respectively.) To find the density of Jupiter compared to Earth, we can use the formula for density, which is mass divided by volume. The volum…
```

### J=5 (WikiText PPL 7.20, prompt PPL 1.29)

```text
____ A. Jupiter B. Saturn C. Venus D. Mars Answer: A The main reason for the large temperature difference between day and night in the Xinjiang region is ____ A. The desert area is large, and the heat retention effect is weak B. The altitude is high, and the solar radiation is weak C. The climate is arid, and the atmospheric weakening effect is weak D. The latitude is high, and the solar altitude …
```

---

## Prompt 1: C — J=8 & J=5 say Paris; J=6 misses it

**Stem:** `The capital of France is`

*Aggregate PPL ranks J=8≈J=6, but routing still flips the first token on some prompts.*

### J=8 (WikiText PPL 6.96, prompt PPL 1.26)

```text
Paris. The capital of Germany is Berlin. The capital of the United States is Washington, D.C. The capital of Brazil is Brasília. The capital of Japan is Tokyo. The capital of South Korea is Seoul. The capital of India is New Delhi. The capital of Egypt is Cairo. The capital of Nigeria is Abuja. The capital of China is Beijing. The capital of Russia is Moscow. The capital of Australia is Canberra. …
```

### J=6 (WikiText PPL 7.06, prompt PPL 1.25)

```text
a well-known fact, but what is the capital of the United Kingdom? The capital of the United Kingdom is London. London is also the capital of England, and is the largest city in the United Kingdom. It is located on the River Thames, and is known for its rich history, cultural landmarks, and iconic buildings such as the Tower of London, Buckingham Palace, and the Houses of Parliament. The city is al…
```

### J=5 (WikiText PPL 7.20, prompt PPL 1.29)

```text
Paris. This is a well-known fact. But what is the capital of the United States? The capital of the United States is Washington, D.C. The capital of the United Kingdom is London. The capital of Brazil is Brasília. The capital of Japan is Tokyo. The capital of China is Beijing. The capital of Canada is Ottawa. The capital of Australia is Canberra. The capital of India is New Delhi. The capital of Eg…
```

---
