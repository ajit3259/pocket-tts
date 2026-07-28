# E15 Generation Review

For each clip, judge intelligibility, exact text coverage, and whether the voice
matches the corresponding fixed E13 prompt.

## overfit_01_female

- Category: `overfit`
- Speaker: `rasa:hindi:female`
- Audio: [overfit_01_female.wav](overfit_01_female.wav)
- Text: बढ़िया। इंजेक्शन का वाणिज्यिक मूल्य क्या है?
- Generated duration: `0.320s`
- Likely hit maximum frames: `False`
- Decision: `rejected`
- Notes: Human review: effectively no audible speech.

## overfit_01_male

- Category: `overfit`
- Speaker: `rasa:hindi:male`
- Audio: [overfit_01_male.wav](overfit_01_male.wav)
- Text: बढ़िया। इंजेक्शन का वाणिज्यिक मूल्य क्या है?
- Generated duration: `0.320s`
- Likely hit maximum frames: `False`
- Decision: `rejected`
- Notes: Human review: only a brief noise at the start, roughly 0.1 seconds.

## unseen_hindi_weather_female

- Category: `control`
- Speaker: `rasa:hindi:female`
- Audio: [unseen_hindi_weather_female.wav](unseen_hindi_weather_female.wav)
- Text: खिड़की के बाहर हल्की बारिश हो रही है और बच्चे आँगन में खेल रहे हैं।
- Generated duration: `9.040s`
- Likely hit maximum frames: `True`
- Decision: `rejected`
- Notes: Human review: sustained e-like sound with no understandable words.

## unseen_hindi_weather_male

- Category: `control`
- Speaker: `rasa:hindi:male`
- Audio: [unseen_hindi_weather_male.wav](unseen_hindi_weather_male.wav)
- Text: खिड़की के बाहर हल्की बारिश हो रही है और बच्चे आँगन में खेल रहे हैं।
- Generated duration: `1.280s`
- Likely hit maximum frames: `False`
- Decision: `rejected`
- Notes: Human review: noise-like output. The user repeated the female path for the fourth observation; this note maps that fourth observation to the male item by sequence.
