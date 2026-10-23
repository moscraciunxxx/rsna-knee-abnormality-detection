# OAI clearance (read 2026-10-12 ~06:30 UTC, browser, logged in as vitaliecervinschi)

Working call until the host answers: do not train on OAI images, and do not treat OAI-trained weights as prize-eligible. Submission 56929063 stays selected. It uses those weights, so it is the public-LB anchor and is not a safe prize pick until the host says yes.

## Rules (competition Rules page, section 2.6)

External Data must be "publicly available and equally accessible to use by all Participants of the Competition for purposes of the competition at no cost to the other Participants," or meet the Reasonableness criteria in 2.6.b.

2.6.b: external data and models are acceptable unless the host prohibits them. Use must be "reasonably accessible to all" and of "minimal cost." The host names "geo restrictions" as a reason access can fail. The host applies a Reasonableness Standard on cost and accessibility. Overview also says: "Freely & publicly available external data is allowed, including pre-trained models."

## Host, in their words

Discussion 733965, reply to the OAI / MRNet / fastMRI+ / SKM-TEA question (posted a month ago): non-commercial license terms do not by themselves exclude a dataset, and prize money is not commercial exploitation. "Datasets that can be accessed through a straightforward registration or click-through data-use agreement would generally satisfy" equal access. "Datasets requiring institution-specific approvals, negotiated written agreements, IRB approval, lengthy credentialing, or other substantial administrative steps may present a meaningful accessibility barrier." The host will not interpret or waive a dataset owner's agreement. Teams must make sure prize use and code/model release are allowed by that agreement.

Discussion 734109 (posted 2 months ago): KneeCoT is not allowed "due to the required data agreement with dataset institution," because that agreement makes the field uneven.

Discussion 741819 (posted 21 days ago), the only host sentence that names OAI: "Using OAI as an external dataset does not violate the RSNA Knee Abnormality Detection challenge rules if it is generally accessible to all participants without an escalated (i.e. institutional) review and sign-off or access to a legal team, and is free of charge." The host will not say whether the OAI stewards allow this competition.

## Still unanswered by the host

- 747455 (3 days ago): are public weights trained on OAI allowed? One non-host comment. No host reply.
- 747660 (3 days ago): do the public 0.949 weights violate the rules? No host reply. Participants disagree on whether OAI is banned.
- 743416 (16 days ago, last comment ~2h before this read): asks yes/no for MRNet, SKM-TEA, fastMRI, OAI, KneeMRI, KneeXNet, and separately for public weights. No host reply. Taeyang (4 days ago) asked the same two questions: OAI via Login.gov, and the public 0.949 checkpoint.
- 748276 (20 hours ago, 0 comments): China cannot register; cites an NIH block since 4 April 2025 on NDA access from countries of concern, including China. Asks if that geo restriction bans OAI under 2.6.b.

Participant comments are not rulings. Tucker Arrants (741819) says OAI is a no because of institutional sign-off. Eesh saxena (743416) says an individual NDA application was approved with no institution and that the Data Permissions page lists "Institutional Approval Required: No." I did not open that logged-in page.

## NDA page, this read

https://nda.nih.gov/oai and https://nda.nih.gov/oai/query-download. Banner: "This repository is under review for potential modification in compliance with Administration directives." Access text: "After registering and obtaining a password, you may browse, download, and use the data for human subjects research for scientific and educational purposes." "To obtain access to OAI data you will need an NDA user account. Account can be requested directly from the NIMH Data Archive (NDA) website." This public page does not say "institutional sign-off." It is not an open download.

## Reply for Vitaly to post

Post as a comment on https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/741819. Do not open a new topic. I will not post it.

---

Two yes/no questions still open after your reply on this thread.

1. OAI images. NDA (https://nda.nih.gov/oai/query-download) requires an NDA account before download, and use is limited to human-subjects research for scientific and educational purposes. Does that account plus the data-use certification count as the escalated institutional sign-off you named here, or as the click-through agreement you said is generally acceptable on discussion 733965? Discussion 748276 also reports that NIH blocks NDA access from countries of concern, including China, which is not on this competition's excluded-country list. Does that geo restriction fail Rule 2.6.b?

2. Weights only. If a team never downloads OAI images, may that team submit a public checkpoint whose author says it was trained on OAI (the CoAtNet in the public 0.949 notebook)? If it places in the prizes, does releasing that checkpoint meet the winner obligation to deliver code and weights?

A yes or no on each is enough.

---
