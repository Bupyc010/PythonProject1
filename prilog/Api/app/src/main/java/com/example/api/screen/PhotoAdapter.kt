package com.example.api.screen

import android.view.LayoutInflater
import android.view.ViewGroup
import androidx.recyclerview.widget.RecyclerView
import com.bumptech.glide.Glide
import com.example.api.databinding.ItemPhotoBinding
import com.example.api.model.PhotoResponse

class PhotoAdapter(private val photos: MutableList<PhotoResponse>) :
    RecyclerView.Adapter<PhotoAdapter.PhotoViewHolder>() {

    class PhotoViewHolder(val binding: ItemPhotoBinding) : RecyclerView.ViewHolder(binding.root)

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): PhotoViewHolder {
        val binding = ItemPhotoBinding.inflate(LayoutInflater.from(parent.context), parent, false)
        return PhotoViewHolder(binding)
    }

    override fun onBindViewHolder(holder: PhotoViewHolder, position: Int) {
        val item = photos[position]
        holder.binding.tvCode.text = item.code

        val url = "http://10.0.2.2:8000/${item.photo}"

        Glide.with(holder.itemView.context)
            .load(url)
            .into(holder.binding.ivPhoto)
    }

    override fun getItemCount() = photos.size

    fun updateData(newPhotos: List<PhotoResponse>) {
        photos.clear()
        photos.addAll(newPhotos)
        notifyDataSetChanged()
    }
}