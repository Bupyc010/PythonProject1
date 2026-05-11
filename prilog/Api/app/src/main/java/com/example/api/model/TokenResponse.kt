package com.example.api.model

import com.google.gson.annotations.SerializedName

data class TokenResponse(
    @SerializedName("access_token")
    val access_token: String,

    @SerializedName("token_type")
    val token_type: String
)