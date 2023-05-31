#pragma once

#include <vector>
#include <utility>
#include <iostream>

template <typename Elem>
class RandomAccessVector
{
public:
    RandomAccessVector(std::size_t initialSize)
        : data(initialSize), count(0)
    { }

    RandomAccessVector& push_back(const Elem& elem)
    {
        if (count < data.size())
            data[count++] = elem;
        else {
            data.push_back(elem);
            ++count;
        }
        return *this;
    }

    Elem remove(const std::size_t index)
    {
        if (index < count)
        {
            std::swap(data[index], data[count - 1]);
            --count;
        }
        return data[count];
    }

    const Elem& operator[](const std::size_t index) const
    {
        return data[index];
    }

    Elem& operator[](const std::size_t index)
    {
        return data[index];
    }

    std::size_t size() const
    {
        return count;
    }

private:
    std::vector<Elem>  data;
    std::size_t        count;
};

