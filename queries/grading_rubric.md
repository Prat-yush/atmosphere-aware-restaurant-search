# Relevance grading rubric

Written 2026-10-20, before any grading evidence was viewed. Grades in `data/eval/pool.csv`
were assigned by Claude (an LLM) following this rubric. A human graded queries t09, t17 and
t24 blind (`data/eval/human_check.csv`); agreement is reported with the results.

## What is judged

Does this restaurant have the atmosphere the query asks for? Food, price and service do not
count. Each query is judged on the topics below. These were written from the query's meaning,
including parts the parser misses (marked †).

## Evidence (and only this)

1. **Yelp attributes**, voted by Yelp users: NoiseLevel (quiet / average / loud / very_loud),
   Ambience (romantic, intimate, classy, upscale, trendy, hipster, divey, touristy, casual),
   Music (live, dj, background_music, ...), good for groups, good for kids, attire, has TV.
2. **Review sentences**: a seeded random sample of up to 4 sentences per topic from all of the
   restaurant's reviews, found with a broad keyword list (see `src/judge.py`). They are not the
   systems' retrieved evidence, and system ranks and scores are never shown to the grader.
3. Name and categories (to recognize, for example, a sports bar or a food stall).

## Grades

| grade | meaning |
| --- | --- |
| 2 | Evidence clearly supports **every** requested topic and nothing important contradicts it. |
| 1 | Supports some requested topics, or the evidence is mixed or weak (for example "quiet at lunch, loud at night"). |
| 0 | Evidence contradicts a requested topic, or there is no evidence for any of them. |

Tie-breakers: attribute votes outweigh a single sentence. Several sentences against one
attribute vote can override it. With no evidence for a topic, that topic counts as unsupported.
Negated topics are judged on the negated meaning ("not too loud" is satisfied by quiet or
average noise).

## Topics per query

| qid | query | topics judged |
| --- | --- | --- |
| t01 | a quiet restaurant | noise: quiet |
| t02 | somewhere with a lively atmosphere | energy: lively |
| t03 | cozy little spot | space: cozy, small |
| t04 | romantic dinner place | romance |
| t05 | dimly lit restaurant | lighting: dim |
| t06 | place with good background music | music: pleasant background music |
| t07 | spacious dining room for a big group | space: spacious; good for groups |
| t08 | restaurant with lots of natural light | lighting: bright, natural light |
| t09 | quiet and romantic for an anniversary dinner | noise: quiet; romance |
| t10 | lively bar with live music | energy: lively; music: live |
| t11 | cozy, dimly lit and intimate | space: cozy; lighting: dim; romance: intimate |
| t12 | fun atmosphere with great music for a birthday | energy: fun; music: good |
| t13 | romantic spot with candlelight and soft music | romance; lighting: candlelight; music: soft |
| t14 | bright, airy and spacious brunch place | lighting: bright†; space: airy, spacious |
| t15 | quiet cozy place to catch up with a friend | noise: quiet; space: cozy |
| t16 | buzzing date night spot | energy: buzzing; romance: date spot |
| t17 | not too loud | noise: quiet or average |
| t18 | a restaurant that isn't noisy | noise: quiet or average |
| t19 | not cramped | space: not cramped |
| t20 | dinner spot where the music isn't too loud | music: not too loud†; noise |
| t21 | a place that's never boring | energy: lively, not dull |
| t22 | not a sports bar | not a sports-bar atmosphere (no TVs and game-day crowds) |
| t23 | no fluorescent lighting | lighting: not harsh or fluorescent (dim or warm) |
| t24 | lively but not loud | energy: lively; noise: quiet or average |
| t25 | quiet but not boring, good for a first date | noise: quiet; energy: not dull; romance: date |
| t26 | cozy but not cramped | space: cozy and not cramped |
| t27 | energetic without loud music | energy: lively; music: not too loud |
| t28 | dim and romantic but not dead | lighting: dim; romance; energy: not dead† |
| t29 | busy and fun but we can still hear each other | energy: busy, fun†; noise: can talk† |
| t30 | intimate but not stuffy | romance: intimate; formality: not stuffy† |
| d01 | peaceful place for a quiet lunch | noise: quiet |
| d02 | romantic with dim lighting | romance; lighting: dim |
| d03 | upbeat place with a good playlist | energy: upbeat; music: good |
| d04 | not too crowded | space: not crowded |
| d05 | a restaurant that isn't dull | energy: not dull |
| d06 | vibrant but not noisy | energy: vibrant; noise: quiet or average |
