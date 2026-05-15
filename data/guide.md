# THINGS Datasets (Concepts, Images & Similarity)

## Overview

The **THINGS** datasets are large-scale resources for studying object concepts, images, and human similarity judgments.

* Repository: OSF
* DOI: `10.17605/OSF.IO/jum2f`

They include:

* **1,854 object concepts**
* **26,107 naturalistic images** (JPEG, up to 1600×1600)

---

# THINGS: Concepts & Images

## What’s Included

* Full image database (~5 GB)
* 53 superordinate categories
* Typicality & nameability ratings
* Concept dimensions (e.g. animacy, size)
* 1,854 **license-free images** (for publications)

---

## How to Download (Recommended: Command Line)

### 1. Install OSF CLI

```bash
pip install osfclient
```

### 2. Download the dataset

```bash
osf -p jum2f clone THINGS-database
```

This will create a local folder called:

```
THINGS-database/
```

---

### 3. Extract the images (IMPORTANT)

The main image archive is password-protected.

#### Step-by-step:

1. Go into the downloaded folder
2. Find the file:

   ```
   password_images.txt
   ```
3. Copy the password inside
4. Extract the images:

```bash
unzip -P YOUR_PASSWORD images_THINGS.zip
```

---

## Alternative: License-Free Images Only

If you only need publication-safe images:

* File: `images_THINGSplus-CC0.zip`
* Size: ~1.18 GB
* No restrictions (CC0 license)

---

## Download via Browser (No CLI)

You can also manually download everything here:

* Go to OSF project page
* Download files individually or as ZIPs

---

## License Notes

* **Original images** → academic use only
* **THINGSplus images (CC0)** → free for publications

---

# THINGSplus (Extended Metadata)

## Overview

**THINGSplus** expands the original dataset with richer annotations for all concepts and images.

---

## What’s Included

* 53 high-level categories (expanded from 27)

* Typicality ratings (all concepts)

* Nameability scores (all images)

* Size ratings

* Additional dimensions:

  * animacy
  * manipulability
  * valence
  * arousal
  * preciousness
  * more

* License-free alternative images for publications

---

## ⬇️ Download

Same repository as THINGS:

```bash
pip install osfclient
osf -p jum2f clone THINGS-database
```

All THINGSplus data is included in that download.

---

# THINGS Similarity Dataset

## Overview

* Repository: OSF
* DOI: `10.17605/OSF.IO/F5RN6`

Large-scale behavioral dataset:

* **4.7 million triplet judgments**
* Task: *“Which object is the odd one out?”*
* Participants: 14,025

---

## What’s Included

* Training set (90%) + test set (10%)
* Triplet indices:

  ```
  concept1, concept2, odd-one-out
  ```
* Participant demographics
* 37,000 repeated triplets (for reliability)
* Related: SPoSE embedding model

---

## How to Download

### Command Line (Recommended)

```bash
pip install osfclient
osf -p f5rn6 clone THINGS-behavior
```

---

### Browser Download

Alternatively, download directly from the OSF interface.