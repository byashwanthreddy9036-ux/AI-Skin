# Presenting the project

Before you start: run `uv run skin-research build-site`, then `uv run skin-research serve-site` and open
http://127.0.0.1:8000 (or open `website/index.html` directly). Use a full-screen browser window
(F11). Every number you mention appears on screen, so you can point to it.

## 5-minute version: the guided tour

Click **▶ Guided tour** (or open `#tour`). Use the arrow keys; each step takes about 40 seconds.

| Step | Show | Say (in your own words) |
|---|---|---|
| 1 · The problem | sharp vs blurred photo of the same melanoma | "AI skin-cancer tools are tested on perfect photos. Real photos are blurry, dark or compressed. Does that matter?" |
| 2 · The dataset | photos per diagnosis | "193 expert-labelled photos the AI had never seen. Mostly ordinary moles, so we score it by its average success per diagnosis." |
| 3 · Transforming | drag the before/after slider | "We created 41 versions of every photo: 8 kinds of damage at 3 strengths, classic enhancements, and damage followed by a fix." |
| 4 · The AI responds | the two sets of probability bars | "Same lesion, same AI. On the sharp photo it says melanoma with high confidence. After a mild blur it says something else." |
| 5 · What we measure | balanced accuracy on originals | "We compare every answer with the true diagnosis, and only count a change as real if its range of plausible values excludes zero." |
| 6 · Results | severe-damage bars | "Most changes made the AI worse, worse damage hurt more, and enhancing good photos never helped." |
| 7 · Why it matters | melanoma detection after mild blur | "A slightly out-of-focus photo made the AI miss most melanomas, often calling them harmless moles, and it didn't become less sure of itself." |

Close with: *"So image quality checks belong in front of any AI skin tool, and AI should be tested on realistic
photos. This is a research experiment with 193 photos and one model, not a diagnostic tool."*

## 10-minute version: tour, then live demo

1. **Tour, steps 1–4** (2 min), as above.
2. **Image Lab** (3 min), the part people remember.
   - Click the **Blur changes the diagnosis** case. Switch to **Before / after slider** and drag it.
   - Point to the **What the AI said** table: prediction, confidence, *Prediction changed*.
   - Click **Strength: mild → moderate → severe** and watch the probability bars shift.
   - Click **Brightness correction rescues it**: damage *and* fix, and the right answer comes back.
   - Click **Enhancement makes it worse**: CLAHE makes the photo look crisper, yet the AI gets it wrong.
3. **Prediction Explorer** (1.5 min). Choose ISIC_0034329 and show the grid of 41 versions with ✓/✗.
   Then scroll to **Confidence is not the same as being right** and read the answer under the
   right-hand chart: wrong answers are often as confident as right ones.
4. **Results** (2.5 min). Read the questions as headings:
   - *How much does each change affect accuracy?* Point to the dashed "original" line and the ▼ marks.
   - *Does more severe damage reduce accuracy?* Eight small charts all slope downwards.
   - *Can a fix recover performance after damage?* Green lines helped and red lines hurt; matching the fix to the damage is what counts.
   - *How well is melanoma detected?* The most important clinical number.
   - Point out the **ISIC 2018 Test** status badge: the larger experiment is shown only once it has
     finished and been verified.
5. **Close** (1 min) with tour step 7 (`#tour7`) and the limitations in **About → Limitations**.

## Likely questions

- **Is this a real AI model?** Yes, a published pretrained EfficientNet-B0 (About → Model architecture),
  loaded at a pinned version; nothing was trained or tuned here.
- **Are the images real?** Yes, every image is an ISIC 2018 photo or the exact transformed copy the AI saw.
  Image IDs are listed in `website/README.md`.
- **How sure are you?** Each result has a 95% confidence interval from 1,000 resamples of whole lesions;
  effects are only called real when the interval excludes zero. The sample is small (193 photos), which
  is why the 1,512-photo test-set experiment is running.
- **Why did enhancement not help?** The model learned from unenhanced photos; enhancement makes images look
  less like what it knows. Only fixes that undo specific damage helped.
- **Can I use it on my skin?** No. It is a research prototype, not a medical device.
