**Image megapixel pricing:**
```cel
sum(outputs.filter(o, o.type=='image').map(o, int(o.resolution_mp * double(prices['per_megapixel']))))
```
