package com.example.api.screen

import android.view.LayoutInflater
import android.view.ViewGroup
import androidx.recyclerview.widget.RecyclerView
import com.example.api.databinding.ItemPhotoBinding
import com.example.api.model.PhotoResponse // Убедись, что путь совпадает с файлом из Шага 1

class PhotoAdapter(private val photos: MutableList<PhotoResponse>) :
    RecyclerView.Adapter<PhotoAdapter.PhotoViewHolder>() {

    class PhotoViewHolder(val binding: ItemPhotoBinding) : RecyclerView.ViewHolder(binding.root)

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): PhotoViewHolder {
        val binding = ItemPhotoBinding.inflate(
            LayoutInflater.from(parent.context),
            parent,
            false
        )
        return PhotoViewHolder(binding)
    }

    override fun onBindViewHolder(holder: PhotoViewHolder, position: Int) {
        val item = photos.getOrNull(position)
        holder.binding.tvCode.text = item?.photo_code ?: "Нет кода"
    }

    override fun getItemCount(): Int = photos.size

    fun updateData(newPhotos: List<PhotoResponse>) {
        photos.clear()
        photos.addAll(newPhotos)
        notifyDataSetChanged()
    }
}