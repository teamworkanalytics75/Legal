# 🚀 Quick Start Guide - Upwork Data Downloader

## 📋 What's Been Done

✅ Created comprehensive OAuth 2.0 authorization flow
✅ Added multiple authorization URL formats to try
✅ Created test script to find working authorization URL
✅ Updated main script with better error handling

---

## 🎯 Next Steps

### Step 1: Test Authorization URLs

Run the test script to see all authorization URL options:

```bash
python3 upwork_oauth_test.py
```

This will show you 6 different URL formats to try.

### Step 2: Try Authorization URLs

1. **Open your browser** and log into Upwork
2. **Try each URL** from the test script (start with Option 1)
3. **Look for**:
   - ✅ Authorization page → Click "Authorize"
   - ✅ Page with authorization code → Copy the code!
   - ❌ "Technical problem" error → Try next URL

### Step 3: Get Authorization Code

After clicking "Authorize", the code should appear:
- **On the page** (for desktop apps with OOB flow)
- **In the URL** (if redirected)

**Example code format**: `ABC123XYZ456` (usually a long alphanumeric string)

### Step 4: Run the Downloader

Once you have the authorization code:

```bash
export UPWORK_AUTH_CODE='your_code_here'
python3 upwork_data_downloader.py
```

Or run the script and paste the code when prompted:

```bash
python3 upwork_data_downloader.py
# Then paste the code when asked
```

---

## 🔧 If Authorization URLs Don't Work

### Option 1: Revoke and Re-Authorize

1. Go to: https://www.upwork.com/ab/account-security/connected-services
2. Find "The Money Machine" (if it exists)
3. Click **Revoke** or **Disconnect**
4. Wait 2-3 minutes
5. Try the authorization URLs again

### Option 2: Check API Key Settings

1. Go to: https://www.upwork.com/ab/account-security/api-keys
2. Click on your API key
3. Verify:
   - Status is **Active**
   - Callback URL is set to `urn:ietf:wg:oauth:2.0:oob` (or empty)
   - Project type is **Desktop**

### Option 3: Try Different Browser

- Clear cache and cookies
- Try incognito/private mode
- Try a different browser

### Option 4: Contact Upwork Support

If nothing works, email: support@upwork.com

**Subject**: OAuth 2.0 Authorization Error - Desktop App

**Include**:
- API Key: `eda38c667ec2afa825d5391bd689f173`
- Error: "Technical problem" when trying to authorize
- What you've tried

---

## 📚 Documentation

- **Complete Solution Guide**: [UPWORK_OAUTH_SOLUTION.md](UPWORK_OAUTH_SOLUTION.md)
- **Test Script**: [upwork_oauth_test.py](upwork_oauth_test.py)
- **Main Script**: [upwork_data_downloader.py](upwork_data_downloader.py)

---

## ✅ Success Checklist

- [ ] Test script shows authorization URLs
- [ ] At least one authorization URL works (shows auth page)
- [ ] Authorization code is obtained
- [ ] Code is exchanged for access token
- [ ] Data download starts successfully

---

**Ready to start?** Run: `python3 upwork_oauth_test.py`

