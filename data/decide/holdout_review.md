reviewed: false

# Holdout review

One row per state; `yes` / `no` / `?` (unsure — 0.5 is the right answer, scored as hedging).
Edit cells in place or tell Claude the id + column + new value. `cat` = s straightforward, n negation, i implicit intent, e extraction-shaped, a ambiguous.

| id | cat | state | complaint | refund | shipping | quality | urgent |
|---|---|---|---|---|---|---|---|
| 1 | s | The pasta was cold and the waiter ignored us. I want my money back. | yes | yes | no | yes | no |
| 2 | s | Package arrived a day early, thanks so much! | no | no | yes | no | no |
| 3 | s | Please refund my order, the jacket doesn't fit and I've already returned it. | no | yes | no | no | no |
| 4 | s | The headphones stopped charging after two weeks. Very disappointed. | yes | no | no | yes | no |
| 5 | s | Where is my parcel? Tracking hasn't moved in nine days. | yes | no | yes | no | no |
| 6 | s | Love the new colour, exactly as pictured. | no | no | no | yes | no |
| 7 | s | Charged twice for the same subscription, please fix it today. | yes | yes | no | no | yes |
| 8 | s | Thanks for the quick reply yesterday. | no | no | no | no | no |
| 9 | s | The zipper broke on day two. Not what I expect at this price. | yes | no | no | yes | no |
| 10 | s | Please cancel and refund, I ordered by mistake. | no | yes | no | no | no |
| 11 | s | My delivery was left in the rain and the box is soaked through. | yes | no | yes | no | no |
| 12 | s | Can you tell me if the blue version is back in stock? | no | no | no | no | no |
| 13 | s | I need this resolved before my flight tomorrow morning. | no | no | no | no | yes |
| 14 | s | The courier marked it delivered but nobody was home and there's no package. | yes | no | yes | no | no |
| 15 | s | Best purchase I've made all year, the build quality is superb. | no | no | no | yes | no |
| 16 | s | I'd like a refund for the damaged mug, photos attached. | yes | yes | no | yes | no |
| 17 | s | How long does standard shipping to Ireland take? | no | no | yes | no | no |
| 18 | s | Your support agent was rude and hung up on me. | yes | no | no | no | no |
| 19 | s | The screen has dead pixels right out of the box. | yes | no | no | yes | no |
| 20 | s | Just wanted to say the onboarding email was really helpful. | no | no | no | no | no |
| 21 | s | Still no refund after three weeks, this is unacceptable. | yes | yes | no | no | yes |
| 22 | s | Could I get an invoice with my company's VAT number on it? | no | no | no | no | no |
| 23 | s | The parcel arrived crushed and two of the four glasses are broken. | yes | no | yes | yes | no |
| 24 | s | Order was delivered to my neighbour by mistake but I have it now, all good. | no | no | yes | no | no |
| 25 | s | The fabric pills after one wash. Cheap. | yes | no | no | yes | no |
| 26 | s | Money back please. The app crashes every time I open it. | yes | yes | no | yes | no |
| 27 | s | I'm locked out of my account and have a presentation in an hour. | no | no | no | no | yes |
| 28 | s | Express shipping was selected and paid for but it took eleven days. | yes | no | yes | no | no |
| 29 | s | The knife is sharp, balanced and feels great in the hand. | no | no | no | yes | no |
| 30 | s | Please stop sending me marketing emails. | yes | no | no | no | no |
| 31 | s | Can I change the delivery address? I move house on Friday. | no | no | yes | no | no |
| 32 | s | The lamp flickers constantly and buzzes. I'd like my money back. | yes | yes | no | yes | no |
| 33 | s | Received the wrong size. I ordered a medium and got an extra large. | yes | no | no | no | no |
| 34 | s | Tracking says it's out for delivery, just checking that's today? | no | no | yes | no | no |
| 35 | s | Everything works perfectly, thank you for the fast turnaround. | no | no | no | no | no |
| 36 | s | The shoes fell apart within a month. Awful quality. | yes | no | no | yes | no |
| 37 | s | Do you offer gift wrapping? | no | no | no | no | no |
| 38 | s | URGENT: the server has been down for 40 minutes and our checkout is broken. | yes | no | no | no | yes |
| 39 | s | The estimated delivery date moved three times. What's going on? | yes | no | yes | no | no |
| 40 | s | I want a full refund, not store credit. | no | yes | no | no | no |
| 41 | s | The coffee maker leaks from the bottom every single time. | yes | no | no | yes | no |
| 42 | s | Super fast shipping, ordered Monday and it was here Wednesday. | no | no | yes | no | no |
| 43 | s | What's your return window for sale items? | no | no | no | no | no |
| 44 | s | This is the fourth email I've sent and nobody has replied. | yes | no | no | no | yes |
| 45 | s | The instructions are unclear and the pieces don't line up. | yes | no | no | yes | no |
| 46 | s | Please process the refund to my original card, not PayPal. | no | yes | no | no | no |
| 47 | s | Item shipped to the wrong country. I'm in Canada, it went to Australia. | yes | no | yes | no | no |
| 48 | s | Just confirming my order went through, I didn't get an email. | no | no | no | no | no |
| 49 | s | The battery life is half what the listing promised. | yes | no | no | yes | no |
| 50 | s | Great customer service, Maria sorted everything in five minutes. | no | no | no | no | no |
| 51 | s | The box was fine but the vase inside was shattered. | yes | no | yes | no | no |
| 52 | s | Need a replacement charger sent overnight, my laptop dies tonight. | no | no | yes | no | yes |
| 53 | s | I've been on hold for an hour. Ridiculous. | yes | no | no | no | no |
| 54 | s | The seams are crooked and there's a loose thread everywhere. | yes | no | no | yes | no |
| 55 | s | Can you refund the shipping fee? The item arrived late. | yes | yes | yes | no | no |
| 56 | s | Which courier do you use for international orders? | no | no | yes | no | no |
| 57 | s | The paint chipped the first time I used it. | yes | no | no | yes | no |
| 58 | s | Thanks, the replacement arrived and works fine. | no | no | yes | no | no |
| 59 | s | I was billed for a plan I cancelled last month. Refund it. | yes | yes | no | no | no |
| 60 | s | Delivered on time and well packaged. | no | no | yes | no | no |
| 61 | s | The smell coming off this rug is unbearable. | yes | no | no | yes | no |
| 62 | s | Do you ship to PO boxes? | no | no | yes | no | no |
| 63 | s | My child's birthday is Saturday and the gift still hasn't shipped. | yes | no | yes | no | yes |
| 64 | s | The stitching on the bag is excellent, very happy. | no | no | no | yes | no |
| 65 | s | I'd like to return this and get my money back, it's not as described. | yes | yes | no | no | no |
| 66 | s | Please update my email address on the account. | no | no | no | no | no |
| 67 | s | The tent leaked during the first night of rain. | yes | no | no | yes | no |
| 68 | s | Shipping cost more than the item itself. Not ordering again. | yes | no | yes | no | no |
| 69 | s | The blender is loud but it does the job well. | no | no | no | yes | no |
| 70 | s | Our whole team is blocked until this license key is fixed. | yes | no | no | no | yes |
| 71 | s | Refund requested for order placed last Tuesday, item never dispatched. | yes | yes | yes | no | no |
| 72 | s | Can you recommend a case for the 13-inch model? | no | no | no | no | no |
| 73 | s | Arrived a week late and the food inside had spoiled. | yes | no | yes | yes | no |
| 74 | s | The chair wobbles and one leg is shorter than the others. | yes | no | no | yes | no |
| 75 | s | Is there a discount for bulk orders? | no | no | no | no | no |
| 76 | s | The driver threw the package over the fence and it landed in the pond. | yes | no | yes | no | no |
| 77 | s | Really impressed with how solid the frame feels. | no | no | no | yes | no |
| 78 | s | Please refund me. Third defective unit in a row. | yes | yes | no | yes | no |
| 79 | s | Just a heads-up that your website's checkout page shows a typo. | no | no | no | no | no |
| 80 | s | I need the tracking number, the parcel was supposed to be here Monday. | yes | no | yes | no | no |
| 81 | s | Deadline is end of day, please escalate. | no | no | no | no | yes |
| 82 | s | The colour is completely different from the photos. Misleading. | yes | no | no | yes | no |
| 83 | s | Thank you for the birthday discount code! | no | no | no | no | no |
| 84 | s | Two items missing from my delivery. | yes | no | yes | no | no |
| 85 | s | This kettle boils faster than my old one and looks great. | no | no | no | yes | no |
| 86 | s | I demand a refund and I'm reporting you to the consumer ombudsman. | yes | yes | no | no | no |
| 87 | s | When will the pre-order ship? | no | no | yes | no | no |
| 88 | s | The customer portal keeps logging me out every two minutes. | yes | no | no | yes | no |
| 89 | s | Smooth checkout, clear emails, no complaints at all. | no | no | no | no | no |
| 90 | s | The mattress sagged in the middle after three months. | yes | no | no | yes | no |
| 91 | s | My package says delivered to the front porch but I don't have a front porch. | yes | no | yes | no | no |
| 92 | s | Store credit is fine, no need for a cash refund. | no | no | no | no | no |
| 93 | s | The surgery is tomorrow and the medical supplies haven't arrived. | yes | no | yes | no | yes |
| 94 | s | Both earbuds sound tinny and the left one cuts out. | yes | no | no | yes | no |
| 95 | s | Would you be able to hold the delivery until I'm back on the 14th? | no | no | yes | no | no |
| 96 | s | Absolutely furious. Wrong item, late, and now no answer from support. | yes | no | yes | no | no |
| 97 | s | Fits perfectly and the material is lovely. | no | no | no | yes | no |
| 98 | s | I want my money back for the extended warranty I never asked for. | yes | yes | no | no | no |
| 99 | s | Please don't leave parcels with the downstairs flat again. | yes | no | yes | no | no |
| 100 | s | Happy with the product, just wondering if spare filters are sold separately. | no | no | no | no | no |
| 101 | n | I don't want a refund, just send a replacement. | ? | no | no | no | no |
| 102 | n | No complaints about the product, but the box was crushed in transit. | ? | no | yes | no | no |
| 103 | n | This isn't urgent, whenever you get a chance. | no | no | no | no | no |
| 104 | n | The delivery was fine; it's the product that's the problem, it stopped working. | yes | no | yes | yes | no |
| 105 | n | Not asking for money back, I just think you should know the strap tore. | yes | no | no | yes | no |
| 106 | n | Nothing wrong with the quality, I simply ordered the wrong size. | no | no | no | yes | no |
| 107 | n | I'm not complaining, but is it normal for the fan to rattle? | ? | no | no | yes | no |
| 108 | n | Keep the money, I don't need a refund, but please fix the bug. | yes | no | no | yes | no |
| 109 | n | Shipping had nothing to do with it, the item was already broken when it was packed. | yes | no | yes | yes | no |
| 110 | n | I never said I wanted a refund. I want the item I paid for. | yes | no | no | no | no |
| 111 | n | It's not that it arrived late, it's that it never arrived at all. | yes | no | yes | no | no |
| 112 | n | Don't rush, no deadline on my side, but the invoice total is wrong. | yes | no | no | no | no |
| 113 | n | I wouldn't call it a complaint, but I won't buy from you again. | yes | no | no | no | no |
| 114 | i | This is the third time I've written in about this. | yes | no | no | no | yes |
| 115 | i | I have guests arriving at six. | no | no | ? | no | yes |
| 116 | i | The wedding is Saturday. | no | no | ? | no | yes |
| 117 | i | I've had to buy a replacement from another shop in the meantime. | yes | ? | ? | no | no |
| 118 | i | My lawyer will be in touch. | yes | no | no | no | no |
| 119 | i | Is this really what you consider acceptable? | yes | no | no | no | no |
| 120 | i | Seven business days, you said. It's been fifteen. | yes | no | yes | no | no |
| 121 | i | I'll take whatever gets this sorted fastest. | no | no | no | no | yes |
| 122 | i | Same problem as last month, and the month before. | yes | no | no | no | no |
| 123 | i | Just put the money back where it came from. | no | yes | no | no | no |
| 124 | i | Every other shop I use manages to get things here in two days. | yes | no | yes | no | no |
| 125 | i | I've stopped using it. It just sits in the drawer. | ? | no | no | yes | no |
| 126 | i | Our store opens in 20 minutes and the card reader won't turn on. | yes | no | no | yes | yes |
| 127 | e | Order #48213 never showed up. | yes | no | yes | no | no |
| 128 | e | Tracking 1Z999AA10123456784 says delivered on the 3rd, nothing here. | yes | no | yes | no | no |
| 129 | e | Invoice INV-2291 was paid twice on the 12th; please return the second payment. | yes | yes | no | no | no |
| 130 | e | Serial K7-4410 has a cracked housing on arrival. | yes | no | no | yes | no |
| 131 | e | Ticket 5582 has been open since March. | yes | no | no | no | yes |
| 132 | e | Order 77120: please hold shipment until the 14th. | no | no | yes | no | no |
| 133 | e | Ref 3391 — the 2 TB model was delivered instead of the 4 TB one I paid for. | yes | no | no | no | no |
| 134 | e | The firmware 4.2.1 update bricked my router last night. | yes | no | no | yes | no |
| 135 | e | Case 10930 closed without a reply. Reopening it. | yes | no | no | no | no |
| 136 | e | Parcel DHL 00340434 has been 'in transit' in Leipzig for two weeks. | yes | no | yes | no | no |
| 137 | e | Please refund transaction TXN-88103 (£42.50) from the 9th. | no | yes | no | no | no |
| 138 | e | Model X200 rev B has the overheating issue your forum thread describes. | yes | no | no | yes | no |
| 139 | a | Still waiting. | ? | no | ? | no | ? |
| 140 | a | Hmm. | ? | no | no | no | no |
| 141 | a | Can I change the colour on my order before it ships? | no | no | ? | no | no |
| 142 | a | The blender works but sounds like a jet engine. Not sure if that's normal. | ? | no | no | yes | no |
| 143 | a | Any update? | ? | no | ? | no | ? |
| 144 | a | Is this the right place to ask about my order? | no | no | no | no | no |
| 145 | a | Interesting choice of packaging. | ? | no | yes | no | no |
| 146 | a | It's fine I guess. | ? | no | no | ? | no |
| 147 | a | So what happens now? | ? | no | no | no | ? |
| 148 | a | Are refunds usually this slow? | ? | ? | no | no | no |
| 149 | a | Did you get my last message? | ? | no | no | no | ? |
| 150 | a | I expected more. | yes | no | no | ? | no |
